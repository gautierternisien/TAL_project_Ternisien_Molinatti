from sklearn.metrics.pairwise import cosine_similarity

def search(query, vectorizer, tfidf_matrix, text_names, top_n=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    results = [(text_names[i], similarities[i]) for i in top_indices]
    return results