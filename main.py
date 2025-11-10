import preprocessing
import search
import tfidf

def main():
    tfidf_matrix, text_names, vectorizer = tfidf.tfidf()
    queries = preprocessing.load_queries('requetes.jsonl')
    correct_pred = 0
    total_pred = 0
    for q in queries:
        ground_truth_file = q['Answer file']
        query1 = preprocessing.clean_text(q['Queries'][0])
        query2 = preprocessing.clean_text(q['Queries'][1])
        total_pred+=2
        results1 = search.search(query1, vectorizer, tfidf_matrix, text_names)
        results2 = search.search(query2, vectorizer, tfidf_matrix, text_names)

        predicted_file1 = results1[0][0]
        predicted_file2 = results2[0][0]

        if predicted_file1 == ground_truth_file:
            correct_pred+=1
            print(f"[Correct] Correct prediction for {query1} 1")
        else:
            print(f"[FAUX] Incorrect prediction for {query1} 1: predicted {predicted_file1}, expected {ground_truth_file}")
        if predicted_file2 == ground_truth_file:
            correct_pred+=1
            print(f"[Correct] Correct prediction for {query2} 2")
        else :
            print(f"[FAUX] Incorrect prediction for {query2} 2: predicted {predicted_file2}, expected {ground_truth_file}")

    accuracy = correct_pred / total_pred if total_pred > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
