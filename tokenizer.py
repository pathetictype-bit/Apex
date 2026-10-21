UNKNOWN_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"


class Tokenizer:
    def __init__(self):
        self.encoding = "character_bpe_v1"
        self.merges = []
        self.tokenized_texts = []
        self.vocab = []
        self.token_to_id_map = {}
        self.id_to_token_map = {}

    def train(self, tokenized_texts, num_merges):
        base_tokens = {
            token
            for tokens in tokenized_texts
            for token in tokens
        }
        learned_tokens = set(base_tokens)

        for _ in range(num_merges):
            pair_counts = count_pairs(tokenized_texts)

            if not pair_counts:
                break

            pair = most_frequent_pairs(pair_counts)
            self.merges.append(pair)
            tokenized_texts = merge_pairs(
                tokenized_texts,
                pair
            )
            learned_tokens.add(pair[0] + pair[1])

        self.tokenized_texts = tokenized_texts
        self.vocab = sorted(learned_tokens)
        self.vocab.extend(
            token
            for token in (UNKNOWN_TOKEN, EOS_TOKEN)
            if token not in learned_tokens
        )

        self.token_to_id_map = {
            token: idx
            for idx, token in enumerate(self.vocab)
        }

        self.id_to_token_map = {
            idx: token
            for idx, token in enumerate(self.vocab)
        }

    def load_state(self, state):
        self.encoding = state.get("encoding", "character_bpe_v1")
        self.merges = [tuple(pair) for pair in state.get("merges", [])]
        self.vocab = list(state.get("vocab", []))
        if UNKNOWN_TOKEN not in self.vocab:
            self.vocab.append(UNKNOWN_TOKEN)
        self.token_to_id_map = {
            token: idx
            for idx, token in enumerate(self.vocab)
        }
        self.id_to_token_map = {
            idx: token
            for idx, token in enumerate(self.vocab)
        }

    def add_special_token(self, token):
        if not isinstance(token, str) or not token:
            raise ValueError("Special tokens must be non-empty strings.")
        if token not in self.token_to_id_map:
            self.vocab.append(token)
            token_id = len(self.vocab) - 1
            self.token_to_id_map[token] = token_id
            self.id_to_token_map[token_id] = token

    def build_vocab(self):
        vocab = set()

        for tokens in self.tokenized_texts:
            vocab.update(tokens)

        return sorted(vocab)

    def tokenize(self, text):
        tokens = list(text)

        for pair in self.merges:
            tokens = merge_pairs([tokens], pair)[0]

        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)

        return [
            self.token_to_id_map.get(token, self.token_to_id_map[UNKNOWN_TOKEN])
            for token in tokens
        ]

    def decode(self, ids):
        tokens = [
            self.id_to_token_map[idx]
            for idx in ids
        ]

        return "".join(tokens)

    def size(self):
        return len(self.vocab)


def load_dataset(filename):
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        dataset = []

        for i in range(0, len(lines), 2):
            input_txt = lines[i]
            target_txt = lines[i + 1]

            dataset.append((input_txt, target_txt))
        return dataset

def target_length_for_prompt(prompt, dataset, tokenizer, default=12):
    for input_text, target_text in dataset:
        if input_text == prompt:
            target_ids = tokenizer.encode(target_text)
            if target_ids:
                return len(target_ids)

    return default

def initialize_tokens(texts):
    return [list(text) for text in texts]

def count_pairs(tokenized_texts):
    pair_counts = {}
    for tokens in tokenized_texts:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if any(character.isspace() for token in pair for character in token):
                continue
            if pair not in pair_counts:
                pair_counts[pair] = 1
            else:
                pair_counts[pair] += 1
    return pair_counts
def most_frequent_pairs(pair_counts):
    return max(pair_counts, key=pair_counts.get)
def merge_pairs(words, pair):
    new_words = []
    for word in words:
        new_word = []
        i = 0
        while i < len(word):
            if (i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]):
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_words.append(new_word)

    return new_words
