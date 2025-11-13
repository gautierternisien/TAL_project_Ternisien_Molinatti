import preprocessing
import glob
import os
import math
import numpy as np


def load():
    data_directory = "wiki_split_extract_2k"
    file_list = glob.glob(os.path.join(data_directory, 'wiki_*.txt'))
    print(f"Found {len(file_list)} files.")
    files = preprocessing.load_files(file_list)
    return files


def build_vocab(texts):
    vocab = set()
    for text in texts:
        tokens = text.split()
        vocab.update(tokens)
    return sorted(list(vocab))


def compute_tf(texts, vocab):
    tf_matrix = []
    for text in texts:
        tokens = text.split()
        token_count = len(tokens)
        tf_row = []
        for word in vocab:
            word_count = tokens.count(word)
            tf_row.append(word_count / token_count if token_count > 0 else 0)
        tf_matrix.append(tf_row)
    return np.array(tf_matrix)


def compute_idf(texts, vocab):
    N = len(texts)
    idf_values = []
    for word in vocab:
        doc_count = sum(1 for text in texts if word in text.split())
        idf_values.append(math.log((N + 1) / (doc_count + 1)) + 1)
    return np.array(idf_values)


def compute_tfidf(tf_matrix, idf_values):
    return tf_matrix * idf_values


class ManualVectorizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}

    def transform(self, texts):
        """Transforme une liste de textes en matrice TF-IDF."""
        if isinstance(texts, str):
            texts = [texts]

        tf_matrix = []
        for text in texts:
            tokens = text.split()
            token_count = len(tokens)
            tf_row = []
            for word in self.vocab:
                word_count = tokens.count(word)
                tf_row.append(word_count / token_count if token_count > 0 else 0)
            tf_matrix.append(tf_row)

        return np.array(tf_matrix)


def tfidf():
    print("Calculating TF-IDF matrix...")
    files = load()
    texts, text_names = files
    cleaned_texts = [preprocessing.clean_text(text) for text in texts]
    print(f"Cleaned texts: {len(cleaned_texts)}")

    vocab = build_vocab(cleaned_texts)
    print(f"Vocabulary size: {len(vocab)}")

    tf_matrix = compute_tf(cleaned_texts, vocab)
    idf_values = compute_idf(cleaned_texts, vocab)
    tfidf_matrix = compute_tfidf(tf_matrix, idf_values)

    vectorizer = ManualVectorizer(vocab)

    print("TF-IDF matrix calculated.")
    return tfidf_matrix, text_names, vectorizer


if __name__ == "__main__":
    tfidf_matrix, text_names, vectorizer = tfidf()
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
