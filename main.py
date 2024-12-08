from __future__ import division
import os
import numpy as np
from math import log10, sqrt
from nltk.util import ngrams
from nltk.corpus import stopwords
from string import punctuation
import nltk
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Download necessary nltk data if not already available
nltk.download('stopwords')
nltk.download('punkt')

# Absolute paths
MODEL = 'trigram'  # Choose 'unigram', 'bigram', or 'trigram'
MEASURE = 'cosine'  # Choose 'cosine' or 'jaccard'
DATASET = 'plagarism-training-set'
SOURCE_FOLDER = os.path.join(DATASET, 'source-document')
SUSPICIOUS_FOLDER = os.path.join(DATASET, 'suspicious-document')

# Debugging paths
print("Source folder:", SOURCE_FOLDER)
print("Suspicious folder:", SUSPICIOUS_FOLDER)

# Get text files from subdirectories
def get_text_files(folder):
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []
    txt_files = []
    for root, _, files in os.walk(folder):
        txt_files.extend(os.path.join(root, f) for f in files if f.endswith('.txt'))
    return txt_files

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
        print(f"Error processing file {file}: {e}")
        return set()

# Preprocess documents using multiprocessing
def preprocess_documents(files, model='unigram', num_workers=None):
    preprocessed_documents = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(preprocess_single_document, file, model): file for file in files}
        for i, future in enumerate(as_completed(futures)):
            doc_words = future.result()
            preprocessed_documents.append(doc_words)
            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                print(f"Preprocessed {i + 1}/{len(files)} documents.")
    return preprocessed_documents

# Compute Document Frequency for a single term
def compute_df_single_term(term, preprocessed_documents):
    return sum(1 for doc_words in preprocessed_documents if term in doc_words)

# Compute Document Frequencies using multiprocessing
def compute_dfs_optimized(unique_words, preprocessed_documents, num_workers=None):
    dfs = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        func = partial(compute_df_single_term, preprocessed_documents=preprocessed_documents)
        futures = {executor.submit(func, term): term for term in unique_words}
        for i, future in enumerate(as_completed(futures)):
            df = future.result()
            dfs.append(df)
            if (i + 1) % 100 == 0 or (i + 1) == len(unique_words):
                print(f"Processed DF for {i + 1}/{len(unique_words)} terms.")
    return dfs

# Compute Inverse Document Frequencies (IDF)
def compute_idfs(num_docs, dfs):
    return [1 + log10(num_docs / df) if df > 0 else 1 for df in dfs]

# Cosine Similarity using sets
def cosine_similarity_set(set1, set2, len1, len2):
    intersection = len(set1 & set2)
    magnitude = sqrt(len1) * sqrt(len2)
    return intersection / magnitude if magnitude else 0

# Jaccard Similarity using sets
def jaccard_similarity_set(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0

# Compare a single suspicious document against all source documents
def compare_single_suspicious(s_index, s_vector, source_vectors, measure):
    similarities = []
    for j, src_vec in enumerate(source_vectors):
        if measure == 'cosine':
            similarity = cosine_similarity_set(src_vec, s_vector, len(src_vec), len(s_vector))
        elif measure == 'jaccard':
            similarity = jaccard_similarity_set(src_vec, s_vector)
        else:
            similarity = 0
        if similarity > 0.0:
            similarities.append((s_index, j, similarity))
    return similarities

# Main Execution
if __name__ == "__main__":
    import time

    start_time = time.time()

    # Load source and suspicious files
    source_files = get_text_files(SOURCE_FOLDER)
    suspicious_files = get_text_files(SUSPICIOUS_FOLDER)

    print(f"Number of source files: {len(source_files)}")
    print(f"Number of suspicious files: {len(suspicious_files)}")

    # Combine all files for vocabulary creation
    all_files = source_files + suspicious_files

    # Preprocess documents into sets of n-grams or words using multiprocessing
    NUM_WORKERS = os.cpu_count() or 4
    print("Preprocessing documents...")
    preprocessed_documents = preprocess_documents(all_files, MODEL, num_workers=NUM_WORKERS)

    # Extract unique terms or n-grams and eliminate stopwords
    print("Extracting unique terms and eliminating stopwords...")
    unique_terms = set().union(*preprocessed_documents)
    unique_terms = eliminate_stopwords(unique_terms)
    unique_terms = sorted(unique_terms)  # Sorting for consistent ordering
    unique_terms_set = set(unique_terms)  # For faster lookup

    print(f"Number of unique terms after eliminating stopwords: {len(unique_terms)}")

    # Compute DF and IDF using multiprocessing
    NUM_DOCS = len(preprocessed_documents)
    print("Computing document frequencies (DF)...")
    dfs = compute_dfs_optimized(unique_terms, preprocessed_documents, num_workers=NUM_WORKERS)

    print("Computing inverse document frequencies (IDF)...")
    idfs = compute_idfs(NUM_DOCS, dfs)

    # Compute TF-IDF vectors as sets (binary vectors)
    print("Computing TF-IDF vectors...")
    tfidf_vectors = [doc_words & unique_terms_set for doc_words in preprocessed_documents]

    # Split TF-IDF vectors into source and suspicious
    source_vectors = tfidf_vectors[:len(source_files)]
    suspicious_vectors = tfidf_vectors[len(source_files):]

    # Compare each suspicious document with all source documents using multiprocessing
    print("Comparing documents...")
    output_file = "similarity_results.txt"

    def process_suspicious(s_idx, s_vec):
        return compare_single_suspicious(s_idx, s_vec, source_vectors, MEASURE)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor, open(output_file, 'w', encoding='utf-8') as out:
        futures = {executor.submit(process_suspicious, i, s_vec): i for i, s_vec in enumerate(suspicious_vectors)}
        for i, future in enumerate(as_completed(futures)):
            similarities = future.result()
            max = 0
            for s_idx, src_idx, sim in similarities:
                if sim > max:
                    max = sim
                #out.write(f"Suspicious doc {s_idx}, Source doc {src_idx}, Similarity: {sim:.4f}\n")
             out.write(f"{max:.4f}\n")
            print(f"Processed {i + 1}/{len(suspicious_vectors)} suspicious documents.")

    print(f"Results written to {output_file}.")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
