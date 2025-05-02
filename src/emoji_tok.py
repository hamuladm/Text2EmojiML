import re
import torch


class EmojiTokenizer:
    def __init__(self, dataset):
        self.emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F"  # Emoticons
            r"\U0001F300-\U0001F5FF"  # Symbols & pictographs
            r"\U0001F680-\U0001F6FF"  # Transport & map symbols
            r"\U0001F700-\U0001F77F"  # Alchemical symbols
            r"\U0001F780-\U0001F7FF"  # Geometric symbols
            r"\U0001F800-\U0001F8FF"  # Supplemental arrows-C
            r"\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
            r"\U0001FA00-\U0001FA6F"  # Chess symbols
            r"\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended-A
            r"\U00002702-\U000027B0"  # Dingbats
            r"\U000024C2-\U0001F251"  # Enclosed characters
            r"]+"
        )
        self.special_tokens = {
            "<pad>": 1,
            "<bos>": 0,
            "<eos>": 2,
        }

        self.pad_token_id = self.special_tokens["<pad>"]
        self.bos = self.special_tokens["<bos>"]
        self.eos = self.special_tokens["<eos>"]

        self.emoji_to_id = {}
        self.id_to_emoji = {}
        self._build_vocab(dataset)

    def _build_vocab(self, dataset):
        unique_emojis = set()
        for text in dataset:
            emojis = self.emoji_pattern.findall(text)
            unique_emojis.update(emojis)

        # Add special tokens to vocab
        self.emoji_to_id = {**self.special_tokens}
        emoji_offset = len(self.special_tokens)
        for idx, emoji in enumerate(sorted(unique_emojis), start=emoji_offset):
            self.emoji_to_id[emoji] = idx

        self.id_to_emoji = {idx: emoji for emoji, idx in self.emoji_to_id.items()}

    def encode(self, emoji_sequence, max_length, add_special_tokens=True):
        emojis = self.emoji_pattern.findall(emoji_sequence)
        encoded = [
            self.emoji_to_id[emoji] for emoji in emojis if emoji in self.emoji_to_id
        ]

        if add_special_tokens:
            encoded = (
                [self.special_tokens["<bos>"]]
                + encoded
                + [self.special_tokens["<eos>"]]
            )

        encoded = self.pad_sequence(encoded, max_length)
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, sequence: torch.Tensor, skip_special_tokens=True):
        emojis = [
            self.id_to_emoji[idx]
            for idx in sequence.tolist()
            if not (skip_special_tokens and idx in self.special_tokens.values())
        ]
        return "".join(emojis)

    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            return sequence + [self.special_tokens["<pad>"]] * (
                max_length - len(sequence)
            )
        else:
            return sequence[:max_length]

    @property
    def vocab_size(self):
        return len(self.emoji_to_id)
