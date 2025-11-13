import preprocessing
import search
import tfidf_hand
import tfidf
import time


def reciprocal_rank(results, ground_truth_file):
    """Retourne 1/rang si ground_truth_file est trouvé dans la liste de résultats, sinon 0.0.
    results: liste triée [(filename, score), ...] du plus pertinent au moins pertinent.
    """
    for idx, (fname, _score) in enumerate(results, start=1):
        if fname == ground_truth_file:
            return 1.0 / idx
    return 0.0


def find_rank(results, ground_truth_file):
    """Retourne le rang (1-indexé) du ground truth s'il est présent dans results, sinon None."""
    for idx, (fname, _score) in enumerate(results, start=1):
        if fname == ground_truth_file:
            return idx
    return None

def is_in_top_k(results, ground_truth_file, k):
    """Vérifie si ground_truth_file est présent dans les k premiers résultats."""
    for fname, _score in results[:k]:
        if fname == ground_truth_file:
            return True
    return False

def main(version):
    tfidf_matrix, text_names, vectorizer = version.tfidf()
    queries = preprocessing.load_queries('requetes.jsonl')
    correct_pred = 0
    total_pred = 0

    k = 5
    correct_at_k = 0

    mrr_sum = 0.0
    num_queries = 0

    for q in queries:
        ground_truth_file = q['Answer file']
        query1 = preprocessing.clean_text(q['Queries'][0])
        query2 = preprocessing.clean_text(q['Queries'][1])
        total_pred+=2
        results1 = search.search(query1, vectorizer, tfidf_matrix, text_names, top_n=k)
        results2 = search.search(query2, vectorizer, tfidf_matrix, text_names, top_n=k)

        # MRR accumulation
        rr1 = reciprocal_rank(results1, ground_truth_file)
        rr2 = reciprocal_rank(results2, ground_truth_file)
        mrr_sum += rr1 + rr2
        num_queries += 2

        # Recall@k : vérifie si le document est dans le top-k
        if is_in_top_k(results1, ground_truth_file, k):
            correct_at_k += 1

        if is_in_top_k(results2, ground_truth_file, k):
            correct_at_k += 1

        predicted_file1 = results1[0][0]
        predicted_file2 = results2[0][0]

        if predicted_file1 == ground_truth_file:
            correct_pred+=1
            print(f"[Correct] Correct prediction for {query1} 1")
        else:
            full_results1 = search.search(query1, vectorizer, tfidf_matrix, text_names, top_n=len(text_names))
            rank1 = find_rank(full_results1, ground_truth_file)
            rank_msg1 = f"rank={rank1}" if rank1 is not None else "rank=absent"
            print(f"[FAUX] Incorrect prediction for {query1} 1: predicted {predicted_file1}, expected {ground_truth_file} ({rank_msg1})")
        if predicted_file2 == ground_truth_file:
            correct_pred+=1
            print(f"[Correct] Correct prediction for {query2} 2")
        else :
            full_results2 = search.search(query2, vectorizer, tfidf_matrix, text_names, top_n=len(text_names))
            rank2 = find_rank(full_results2, ground_truth_file)
            rank_msg2 = f"rank={rank2}" if rank2 is not None else "rank=absent"
            print(f"[FAUX] Incorrect prediction for {query2} 2: predicted {predicted_file2}, expected {ground_truth_file} ({rank_msg2})")

    accuracy = correct_pred / total_pred if total_pred > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")

    recall_at_k = correct_at_k / total_pred if total_pred > 0 else 0
    print(f"Recall@{k}: {recall_at_k:.2%}")

    # MRR computation and display
    mrr = (mrr_sum / num_queries) if num_queries > 0 else 0.0
    print(f"MRR@{k}: {mrr:.4f}")


if __name__ == "__main__":
    print("Utilisation de TF-IDF codé à la main:")
    start_time = time.time()
    version = tfidf_hand
    main(version)
    hand_time = time.time() - start_time
    print(f"Temps d'exécution TF-IDF manuel: {hand_time:.2f}s")

    print("\nUtilisation de TF-IDF de sklearn:")
    start_time = time.time()
    version = tfidf
    main(version)
    sklearn_time = time.time() - start_time
    print(f"Temps d'exécution TF-IDF sklearn: {sklearn_time:.2f}s")

