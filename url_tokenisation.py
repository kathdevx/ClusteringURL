import tensorflow as tf


def create_vocuabulary(urls):
    """Create a character vocabulary

    Args:
        urls (list[str]): list of urls

    Notes:
        enumerate starts from 1 since 0 value is left for padding

    Returns:
        characters (list[str]): list of unique characters found in urls
        char_to_idx (dict): dictionary that maps character to id
    """
    characters = sorted(list(set("".join(urls))))
    char_to_idx = {ch: i for i, ch in enumerate(characters, start=1)}
    idx_to_char = {i: ch for i, ch in enumerate(characters, start=1)}
    return characters, char_to_idx


def url_to_sequence(urls, char_to_idx):
    """Convert URLs to character-level sequences

    Args:
        urls (list[str]): list of urls
        char_to_idx: char_to_idx (dict): dictionary that maps character to id

    Returns:
        sequences (list[int]): a list of sequences that transform the characters in the url
                               to the appropriate index (int)
    """
    sequences = []
    for url in urls:
        sequence = [char_to_idx[ch] for ch in url]
        sequences.append(sequence)
    return sequences


def pad_sequences(sequences, length):
    """Pad sequences to a fixed length

    Args:
        sequences (list[int]): a list of sequences that transform the characters in the url
                               to the appropriate index (int)
        length (int): length of sequence to add padding in the end

    Returns:
        padded_sequences (list[int])
    """
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=length, padding='post'
    )
    return padded_sequences


def create_embedding_model(characters, length):
    """Define the character-level embedding model

    Args:
        characters (list[str]): list of unique characters found in urls
        length (int):

    Returns:
        model (Keras Model): keras model that creates embeddings
    """
    embedding_dim = 100
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(characters) + 1, embedding_dim, input_length=length),
        tf.keras.layers.GlobalAveragePooling1D()
    ])
    return model


def print_url_with_embeddings(urls, embeddings):
    """Print the character-level embeddings

    Args:
        urls (list[str]): list of urls
        embeddings (array): embedding for each url

    Returns:

    """
    for i, url in enumerate(urls, start=1):
        print(f'URL: {url}')
        print(f'Embedding: {embeddings[i]}')
        print('=' * 100)


def calculate_max_length_of_sequences(sequences):
    """Calculate the max length of a sequence in the dataset

    Args:
        sequences (list[int]): a list of sequences that transform the characters in the url
                               to the appropriate index (int)

    Returns:
        max_length (int): max length of sequences
    """
    max_length = max(len(seq) for seq in sequences)
    return max_length
