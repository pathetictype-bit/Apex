from Charecter_library import library

SPECIAL_TOKENS = ["|", "~"] 
def encode(char):
    return library[char]

def encode_txt(text):
    vectors = []
    i = 0

    while i < len(text):

        # Check for special tokens first
        matched = False

        for token in SPECIAL_TOKENS:
            if text.startswith(token, i):
                vectors.append(encode(token))
                i += len(token)
                matched = True
                break

        if matched:
            continue

        # Otherwise encode one character
        vectors.append(encode(text[i]))
        i += 1

    return vectors