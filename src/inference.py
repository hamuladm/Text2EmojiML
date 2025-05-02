import torch
from transformers import AutoTokenizer
from transformer import TransformerTranslator
import warnings

warnings.filterwarnings("ignore")

# Hyperparameters
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

embed_dim = 128
num_blocks = 2
num_heads = 2
max_context_length = 64

MODEL_CHECKPOINT_PATH = "weights/transformer.49.pth"
TEXT_TOKENIZER_PATH = "distilroberta-base"


def load_model_and_tokenizers():
    # Load tokenizers
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_TOKENIZER_PATH)
    emoji_tokenizer = AutoTokenizer.from_pretrained(TEXT_TOKENIZER_PATH)

    # Initialize model
    encoder_vocab_size = text_tokenizer.vocab_size
    output_vocab_size = emoji_tokenizer.vocab_size

    model = TransformerTranslator(
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        encoder_vocab_size=encoder_vocab_size,
        output_vocab_size=output_vocab_size,
        CUDA=CUDA,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])

    return model, text_tokenizer, emoji_tokenizer


def predict_emoji(text, model, text_tokenizer, emoji_tokenizer):
    model.eval()

    # Tokenize input text
    text_tokens = text_tokenizer(
        text,
        truncation=True,
        max_length=max_context_length,
        padding="max_length",
        return_tensors="pt",
    ).input_ids.to(device)

    model.encode(text_tokens)
    start_token = emoji_tokenizer.bos_token_id
    output_tokens = torch.tensor([[start_token]], device=device)

    # Generate tokens one by one
    for _ in range(max_context_length - 1):
        with torch.no_grad():
            logits = model(output_tokens)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        output_tokens = torch.cat((output_tokens, next_token), dim=1)
        if next_token.item() == emoji_tokenizer.eos_token_id:
            break

    emoji_sequence = emoji_tokenizer.decode(output_tokens.squeeze())
    return emoji_sequence


def main():
    print("Loading model and tokenizers...")
    model, text_tokenizer, emoji_tokenizer = load_model_and_tokenizers()

    while True:
        text = input("Input Text: ").strip()
        if text.lower() == "exit":
            exit(0)

        emoji_sequence = predict_emoji(text, model, text_tokenizer, emoji_tokenizer)
        print(f"Predicted Emoji: {emoji_sequence}\n")


if __name__ == "__main__":
    main()
