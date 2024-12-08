from __future__ import division
import multiprocessing
import re
import nltk
import os
import numpy as np
from math import log10, sqrt
from nltk.util import ngrams
from nltk.corpus import stopwords
from string import punctuation

# Download stopwords if not already available
nltk.download('stopwords')

# Paths relative to the current script directory
MODEL = 'trigram'  # Choose 'unigram', 'bigram', or 'trigram'
MEASURE = 'cosine'  # Choose 'cosine' or 'jaccard'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FOLDER = os.path.join(BASE_DIR, 'external-detection-corpus-training', 'source-document')
SUSPICIOUS_FOLDER = os.path.join(BASE_DIR, 'external-detection-corpus-training', 'suspicious-document')

# Debugging paths
print("Source folder:", SOURCE_FOLDER)
print("Suspicious folder:", SUSPICIOUS_FOLDER)

# Get text files
def get_text_files(folder):
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt') and os.path.isfile(os.path.join(folder, f))]

# Remove punctuations from text
def remove_punctuation(text):
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)

# Extract n-grams from text (unigram, bigram, trigram)
def extract_ngrams(text, n=1):
    words = remove_punctuation(text.lower()).split()
    ngrams_list = list(ngrams(words, n))
    return [' '.join(gram) for gram in ngrams_list]

# Preprocess documents into sets of words or n-grams
def preprocess_documents(file, model='unigram'):
    with open(file, 'r', encoding='utf-8') as f:
        text = remove_punctuation(f.read().lower())
        if model == 'bigram':
            doc_ngrams = set(extract_ngrams(text, 2))
        elif model == 'trigram':
            doc_ngrams = set(extract_ngrams(text, 3))
        else:
            doc_ngrams = set(text.split())
    return doc_ngrams

# Multiprocess documents
def preprocess_documents_multi(files, model='unigram'):
    print("Files to be processed:", files)  # Debugging
    with multiprocessing.Pool() as pool:
        preprocessed_docs = pool.starmap(preprocess_documents, [(file, model) for file in files])
    return preprocessed_docs

# Compute Document Frequencies (DF)
def compute_dfs_optimized(unique_words, preprocessed_documents):
    dfs = []
    for i, word in enumerate(unique_words):
        if i % 100 == 0:
            print(f"Processing DF for word {i}/{len(unique_words)}")
        count = sum(1 for doc_words in preprocessed_documents if word in doc_words)
        dfs.append(count)
    return dfs

# Compute Inverse Document Frequencies (IDF)
def compute_idfs(num_docs, dfs):
    return [1 + log10(num_docs / df) if df > 0 else 1 for df in dfs]

# Compute TF-IDF Weight Vector
def compute_tfidf_vector(preprocessed_doc, unique_words, idfs):
    return [1 if word in preprocessed_doc else 0 for word in unique_words]

# Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = sqrt(sum(v**2 for v in vec1)) * sqrt(sum(v**2 for v in vec2))
    return dot_product / magnitude if magnitude else 0

# Jaccard Similarity
def jaccard_similarity(vec1, vec2):
    intersection = sum(min(v1, v2) for v1, v2 in zip(vec1, vec2))
    union = sum(max(v1, v2) for v1, v2 in zip(vec1, vec2))
    return intersection / union if union else 0

# Remove stopwords
def eliminate_stopwords(words):
    stop_words = set(stopwords.words('english'))
    return [word for word in words if not any(w in stop_words for w in word.split())]

# Function to compare vectors and write output
def checker(args):
    (i, s_vec), (j, src_vec) = args
    if MEASURE == 'cosine':
        similarity = cosine_similarity(s_vec, src_vec)
    elif MEASURE == 'jaccard':
        similarity = jaccard_similarity(s_vec, src_vec)
    if similarity > 0.0:
        out.write(f"Suspicious doc {i}, Source doc {j}, Similarity: {similarity:.4f}\n")
    if (i * len(source_vectors) + j) % 10 == 0:
        print(f"Processed comparison {i * len(source_vectors) + j}")

# Main Execution
if __name__ == "__main__":
    # Load source and suspicious files
    source_files = get_text_files(SOURCE_FOLDER)
    suspicious_files = get_text_files(SUSPICIOUS_FOLDER)

    # Combine all files for vocabulary creation
    all_files = source_files + suspicious_files

    # Preprocess documents into sets of n-grams or words
    print("Preprocessing documents...")
    preprocessed_documents = preprocess_documents_multi(all_files, MODEL)

    # Extract unique terms or n-grams
    unique_terms = set()
    for doc_words in preprocessed_documents:
        unique_terms.update(doc_words)
    unique_terms = eliminate_stopwords(unique_terms)

    # Compute DF and IDF
    NUM_DOCS = len(preprocessed_documents)
    print("Computing document frequencies (DF)...")
    dfs = compute_dfs_optimized(unique_terms, preprocessed_documents)
    print("Computing inverse document frequencies (IDF)...")
    idfs = compute_idfs(NUM_DOCS, dfs)

    # Compute TF-IDF vectors
    print("Computing TF-IDF vectors...")
    tfidf_vectors = [compute_tfidf_vector(doc_words, unique_terms, idfs) for doc_words in preprocessed_documents]

    # Compare each suspicious document with all source documents
    print("Comparing documents...")
    output_file = "similarity_results.txt"
    with open(output_file, 'w', encoding='utf-8') as out:
        suspicious_vectors = tfidf_vectors[len(source_files):]
        source_vectors = tfidf_vectors[:len(source_files)]

        with multiprocessing.Pool() as pool:
            args = [((i, s_vec), (j, src_vec)) for i, s_vec in enumerate(suspicious_vectors) for j, src_vec in enumerate(source_vectors)]
            pool.map(checker, args)

    print(f"Processing complete. Results written to {output_file}.")
