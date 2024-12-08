from __future__ import division
import os
import sys
import numpy as np
from math import log10, sqrt
from nltk.util import ngrams
from nltk.corpus import stopwords
from string import punctuation
import nltk
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from collections import Counter

# Remove punctuations from text
def remove_punctuation(text):
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)

# Extract n-grams from text (unigram, bigram, trigram)
def extract_ngrams(text, n=1):
    words = remove_punctuation(text.lower()).split()
    return [' '.join(gram) for gram in ngrams(words, n)]

# Remove stopwords
def eliminate_stopwords(words):
    stop_words = set(stopwords.words('english'))
    return {word for word in words if not any(w in stop_words for w in word.split())}

# Preprocess a single document
def preprocess_single_document(file, model='unigram'):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            text = remove_punctuation(f.read().lower())
            if model == 'bigram':
                doc_ngrams = set(extract_ngrams(text, 2))
            elif model == 'trigram':
                doc_ngrams = set(extract_ngrams(text, 3))
            else:
                doc_ngrams = set(text.split())
        return doc_ngrams
    except Exception as e:
        print(f"Error processing file {file}: {e}", file=sys.stderr)
        return set()

# Compare a single suspicious document against all source documents
def compare_single_suspicious(args):
    s_idx, s_vec, source_vectors, measure = args
    similarities = []
    len_s = len(s_vec)
    for j, src_vec in enumerate(source_vectors):
        len_src = len(src_vec)
        if measure == 'cosine':
            intersection = len(s_vec & src_vec)
            magnitude = sqrt(len_src) * sqrt(len_s)
            similarity = intersection / magnitude if magnitude else 0
        elif measure == 'jaccard':
            intersection = len(s_vec & src_vec)
            union = len(s_vec | src_vec)
            similarity = intersection / union if union else 0
        else:
            similarity = 0
        if similarity > 0.0:
            similarities.append((s_idx, j, similarity))
    return similarities

# Get text files
def get_text_files(folder):
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]
    print(f"Found {len(files)} '.txt' files in '{folder}'.")
    return files

# Compute Document Frequencies using a single pass with Counter
def compute_dfs(preprocessed_documents):
    print("Computing document frequencies (DF) using Counter...")
    df_counter = Counter()
    for i, doc_words in enumerate(preprocessed_documents, 1):
        df_counter.update(doc_words)
        if i % 100 == 0 or i == len(preprocessed_documents):
            print(f"Updated DF for {i}/{len(preprocessed_documents)} documents.")
    return df_counter

# Compute Inverse Document Frequencies (IDF)
def compute_idfs(num_docs, df_counter):
    print("Computing inverse document frequencies (IDF)...")
    idfs = {}
    for term, df in df_counter.items():
        if df > 0:
            idfs[term] = 1 + log10(num_docs / df)
        else:
            idfs[term] = 1
    return idfs

# Compute TF-IDF Weight Vector (binary)
def compute_tfidf_vector(preprocessed_doc, unique_terms_set):
    return {word for word in preprocessed_doc if word in unique_terms_set}

# Main Execution
def main():
    import time

    start_time = time.time()

    # Absolute paths
    MODEL = 'trigram'  # Choose 'unigram', 'bigram', or 'trigram'
    MEASURE = 'cosine'  # Choose 'cosine' or 'jaccard'
    DATASET = 'external-detection-corpus-training'
    SOURCE_FOLDER = os.path.join(DATASET, 'source-document')
    SUSPICIOUS_FOLDER = os.path.join(DATASET, 'suspicious-document')

    # Debugging paths
    print("\n=== Plagiarism Checker Initialization ===")
    print(f"Source folder: {SOURCE_FOLDER}")
    print(f"Suspicious folder: {SUSPICIOUS_FOLDER}")

    # Download stopwords if not already available
    print("\nEnsuring NLTK stopwords are downloaded...")
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    else:
        print("NLTK stopwords already downloaded.")

    # Load source and suspicious files
    print("\nLoading source and suspicious files...")
    source_files = get_text_files(SOURCE_FOLDER)
    suspicious_files = get_text_files(SUSPICIOUS_FOLDER)

    print(f"Number of source files: {len(source_files)}")
    print(f"Number of suspicious files: {len(suspicious_files)}")
    print(f"Total files to process: {len(source_files) + len(suspicious_files)}")

    if not source_files and not suspicious_files:
        print("Error: No files found to process. Please ensure that the source and suspicious folders contain '.txt' files.")
        sys.exit(1)

    # Combine all files for vocabulary creation
    all_files = source_files + suspicious_files

    # Preprocess documents into sets of n-grams or words using multiprocessing
    print("\nPreprocessing documents...")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(preprocess_single_document, file, MODEL): file for file in all_files}
        preprocessed_documents = []
        for i, future in enumerate(as_completed(futures), 1):
            doc_words = future.result()
            preprocessed_documents.append(doc_words)
            if i % 50 == 0 or i == len(all_files):
                print(f"Preprocessed {i}/{len(all_files)} documents.")

    if not preprocessed_documents:
        print("Error: Preprocessing resulted in no documents. Exiting.")
        sys.exit(1)

    # Extract unique terms or n-grams and eliminate stopwords
    print("\nExtracting unique terms and eliminating stopwords...")
    unique_terms = set().union(*preprocessed_documents)
    print(f"Number of unique terms before eliminating stopwords: {len(unique_terms)}")
    unique_terms = eliminate_stopwords(unique_terms)
    unique_terms = sorted(unique_terms)  # Sorting for consistent ordering
    unique_terms_set = set(unique_terms)  # For faster lookup

    print(f"Number of unique terms after eliminating stopwords: {len(unique_terms)}")

    if not unique_terms:
        print("Error: No unique terms found after eliminating stopwords. Exiting.")
        sys.exit(1)

    # Compute DF using a single pass with Counter
    NUM_DOCS = len(preprocessed_documents)
    df_counter = compute_dfs(preprocessed_documents)

    if not df_counter:
        print("Error: Document frequencies computation failed. Exiting.")
        sys.exit(1)

    # Compute IDF
    idfs = compute_idfs(NUM_DOCS, df_counter)

    if not idfs:
        print("Error: Inverse document frequencies computation failed. Exiting.")
        sys.exit(1)

    # Compute TF-IDF vectors as sets (binary vectors)
    print("\nComputing TF-IDF vectors...")
    tfidf_vectors = []
    for i, doc_words in enumerate(preprocessed_documents, 1):
        tfidf_vector = compute_tfidf_vector(doc_words, unique_terms_set)
        tfidf_vectors.append(tfidf_vector)
        if i % 100 == 0 or i == len(preprocessed_documents):
            print(f"Computed TF-IDF for {i}/{len(preprocessed_documents)} documents.")

    # Split TF-IDF vectors into source and suspicious
    source_vectors = tfidf_vectors[:len(source_files)]
    suspicious_vectors = tfidf_vectors[len(source_files):]

    # Prepare arguments for multiprocessing
    comparison_args = [
        (i, s_vec, source_vectors, MEASURE)
        for i, s_vec in enumerate(suspicious_vectors)
    ]

    # Compare each suspicious document with all source documents using multiprocessing
    print("\nComparing documents...")
    output_file = "similarity_results.txt"

    total_similarities = 0
    processed_suspicious = 0

    with ProcessPoolExecutor() as executor, open(output_file, 'w', encoding='utf-8') as out:
        futures = {executor.submit(compare_single_suspicious, arg): arg[0] for arg in comparison_args}
        for future in as_completed(futures):
            s_idx = futures[future]
            try:
                similarities = future.result()
                for s_idx, src_idx, sim in similarities:
                    out.write(f"Suspicious doc {s_idx}, Source doc {src_idx}, Similarity: {sim:.4f}\n")
                    total_similarities += 1
            except Exception as e:
                print(f"Error comparing suspicious doc {s_idx}: {e}", file=sys.stderr)
            processed_suspicious += 1
            if processed_suspicious % 5 == 0 or processed_suspicious == len(suspicious_vectors):
                print(f"Processed {processed_suspicious}/{len(suspicious_vectors)} suspicious documents.")

    print(f"\nProcessing complete. Total similarities found: {total_similarities}")
    print(f"Results written to '{output_file}'.")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

# Define compare_single_suspicious at top level
def compare_single_suspicious(args):
    s_idx, s_vec, source_vectors, measure = args
    similarities = []
    len_s = len(s_vec)
    for j, src_vec in enumerate(source_vectors):
        len_src = len(src_vec)
        if measure == 'cosine':
            intersection = len(s_vec & src_vec)
            magnitude = sqrt(len_src) * sqrt(len_s)
            similarity = intersection / magnitude if magnitude else 0
        elif measure == 'jaccard':
            intersection = len(s_vec & src_vec)
            union = len(s_vec | src_vec)
            similarity = intersection / union if union else 0
        else:
            similarity = 0
        if similarity > 0.0:
            similarities.append((s_idx, j, similarity))
    return similarities

# Ensure that compare_single_suspicious is defined before main
if __name__ == "__main__":
    main()
