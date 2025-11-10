import os
import json
import spacy
from spacy.cli import download as spacy_download

def load_spacy_model(name="fr_core_news_sm"):
    try:
        return spacy.load(name)
    except OSError:
        spacy_download(name)
        return spacy.load(name)

nlp = load_spacy_model()

def load_files(file_paths):
    text_names = []
    texts = []
    for path in file_paths:
        text_names.append(os.path.basename(path))
        with open(path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts, text_names

def clean_text(text):
    doc = nlp(text)
    cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return ' '.join(cleaned_tokens) # pour avoir un string

def load_queries(file_path):
    queries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                queries.append(json.loads(line))
    return queries