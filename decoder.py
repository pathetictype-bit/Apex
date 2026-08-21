from Charecter_library import library

index_to_character = {
    index: char
    for index, char in enumerate(library)
}

def decode(output):
    predicted_index = output.index(max(output))
    return index_to_character[predicted_index]
