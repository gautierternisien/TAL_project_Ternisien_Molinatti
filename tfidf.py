import preprocessing
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf():
    data_directory = "wiki_split_extract_2k"
    file_list = glob.glob(os.path.join(data_directory, 'wiki_*.txt'))
    files = preprocessing.load_files(file_list)
    texts, text_names = files
    cleaned_texts = [preprocessing.clean_text(text) for text in texts]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    return tfidf_matrix, text_names, vectorizer