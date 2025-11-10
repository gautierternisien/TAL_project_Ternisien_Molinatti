import preprocessing
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf():
    print("Calculating TF-IDF matrix...")
    data_directory = "wiki_split_extract_2k"
    file_list = glob.glob(os.path.join(data_directory, 'wiki_*.txt'))
    print(f"Found {len(file_list)} files.")
    files = preprocessing.load_files(file_list)
    print("Files loaded.")
    texts, text_names = files
    cleaned_texts = [preprocessing.clean_text(text) for text in texts]
    print(f"Cleaned texts: {len(cleaned_texts)}")

    #de base ngram_range = (1, 1) mais (1,2) permet d'avoir de meilleures perf
    #de base strip_accents = None, le passer à strip_accents='unicode' permet de passer d'une accuracy de 81% à 82%
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    print("TF-IDF matrix calculated.")
    return tfidf_matrix, text_names, vectorizer