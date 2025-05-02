import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from transformer import TransformerTranslator

import warnings

warnings.filterwarnings("ignore")

DEBUG = 0

# Hyperparameters
CUDA = torch.cuda.is_available()
VALIDATE_AMOUNT = 10

batch_size = 32
embed_dim = 128
num_blocks = 2
num_heads = 2
num_epochs = 50
learning_rate = 1e-3

device = torch.device("cuda" if CUDA else "cpu")


# Dataset build
class Text2EmojiDataset(Dataset):
    def __init__(self, dataset, text_tokenizer, emoji_tokenizer, max_length=32):
        self.dataset = dataset
        self.text_tokenizer = text_tokenizer
        self.emoji_tokenizer = emoji_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset[idx]["text"] is None or self.dataset[idx]["emoji"] is None:
            return {
                "text": torch.zeros(self.max_length, dtype=torch.long),
                "emoji": torch.zeros(self.max_length, dtype=torch.long),
                "logit_mask": torch.zeros(self.max_length, dtype=torch.long),
            }

        # Tokenize text
        text_tokens = self.text_tokenizer(
            self.dataset[idx]["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.squeeze()

        # Tokenize emoji
        emoji_tokens = self.emoji_tokenizer(
            self.dataset[idx]["emoji"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.squeeze()

        # Create logit mask
        logit_mask = (text_tokens != self.text_tokenizer.pad_token_id).long()
        if DEBUG:
            print(self.dataset[idx]["text"], self.dataset[idx]["emoji"])
            print(text_tokens, emoji_tokens, logit_mask, sep="\n")
            print(text_tokens.shape, emoji_tokens.shape, logit_mask.shape)

        return {"text": text_tokens, "emoji": emoji_tokens, "logit_mask": logit_mask}

    def logit_to_sentence(self, logits):
        tokens = torch.argmax(logits, dim=-1)
        return self.emoji_tokenizer.decode(tokens)


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def main():
    # Load dataset
    dataset = load_dataset("KomeijiForce/Text2Emoji", split="train[:1%]")

    # Load tokenizers
    text_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    emoji_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Create dataset and dataloader
    full_dataset = Text2EmojiDataset(dataset, text_tokenizer, emoji_tokenizer)

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=True)

    print(text_tokenizer.vocab_size)
    print(emoji_tokenizer.vocab_size)
    encoder_vocab_size = text_tokenizer.vocab_size
    output_vocab_size = emoji_tokenizer.vocab_size

    # Model
    model = TransformerTranslator(
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        encoder_vocab_size=encoder_vocab_size,
        output_vocab_size=output_vocab_size,
        CUDA=CUDA,
    ).to(device)

    print("Number of parameters: ", numel(model))
    print("Number of parameters (trainable): ", numel(model, only_trainable=True))

    # Loss Function + Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=text_tokenizer.pad_token_id)

    # Training Loop
    train_losses = []
    test_losses = []
    num_steps = 0

    for epoch in range(num_epochs):
        running_loss = []
        running_test_loss = []

        model.train()

        for _, batch in enumerate(tqdm(dataloader_train)):
            model.zero_grad()
            optimizer.zero_grad()

            text = batch["text"].to(device)
            emoji = batch["emoji"].to(device)
            logit_mask = batch["logit_mask"].to(device)

            model.encode(text)
            all_outs = torch.tensor([], requires_grad=True).to(device)
            all_outs_tokens = emoji[:, :1]

            for i in range(emoji.shape[1] - 1):
                out = model(all_outs_tokens[:, : i + 1])
                all_outs = torch.cat((all_outs, out), dim=1)
                out_token = torch.argmax(out, dim=2)
                all_outs_tokens = torch.cat((all_outs_tokens, out_token), dim=1)

            all_outs = all_outs * logit_mask[:, 1:, None]
            emoji_masked = emoji[:, 1:] * logit_mask[:, 1:]

            loss = criterion(
                all_outs.view(
                    -1, output_vocab_size
                ),  # (batch_size * max_length, vocab_size)
                emoji_masked.reshape(-1).type(torch.int64),  # (batch_size * max_length)
            )

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            num_steps += 1

        # Validation
        model.eval()

        with torch.no_grad():
            for j, batch in enumerate(dataloader_val):
                text = batch["text"].to(device)
                emoji = batch["emoji"].to(device)
                logit_mask = batch["logit_mask"].to(device)

                if DEBUG:
                    print(text.shape, emoji.shape, logit_mask.shape)

                model.encode(text)
                all_outs = torch.tensor([], requires_grad=False).to(device)
                all_outs_tokens = emoji[:, :1]

                for i in range(emoji.shape[1] - 1):
                    out = model(all_outs_tokens[:, : i + 1])
                    out_token = torch.argmax(out, dim=2)
                    all_outs = torch.cat((all_outs, out), dim=1)
                    all_outs_tokens = torch.cat((all_outs_tokens, out_token), dim=1)

                all_outs = all_outs * logit_mask[:, 1:, None]
                emoji_masked = emoji[:, 1:] * logit_mask[:, 1:]

                loss = criterion(
                    all_outs.view(-1, output_vocab_size),
                    emoji_masked.reshape(-1),
                )

                running_test_loss.append(loss.item())
                print(
                    "Predicted emoji:    ", full_dataset.logit_to_sentence(all_outs[0])
                )
                print("Ground truth Emoji: ", emoji_tokenizer.decode(emoji.squeeze()))

                if j == VALIDATE_AMOUNT:
                    break

        avg_test_loss = np.array(running_test_loss).mean()
        test_losses.append(avg_test_loss)
        avg_loss = np.array(running_loss).mean()
        train_losses.append(avg_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}; Train loss: {avg_loss:.2f}; Test loss: {avg_test_loss:.2f}"
        )

        # Save checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_steps": num_steps,
                "train_losses": train_losses,
                "test_losses": test_losses,
            },
            f"transformer.{epoch}.pth",
        )


if __name__ == "__main__":
    main()
