import string
tokens = list(string.ascii_lowercase) + [" ", "?", ",", "!", ".", "'", "|", "~" ]
library = {}
for i, token in enumerate(tokens):
    vector = [0] * len(tokens)
    vector[i] = 1
    library[token] = vector