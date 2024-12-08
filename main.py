from __future__ import division
import os
import numpy as np
from math import log10, sqrt
from string import punctuation
import nltk
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Download stopwords if not already available
nltk.download('stopwords')

# Configuration Constants
MODEL = 'trigram'        # Options: 'unigram', 'bigram', 'trigram'
MEASURE = 'cosine'       # Options: 'cosine', 'jaccard'
DATASET = 'training-corpus'
SOURCE_FOLDER = os.path.join(DATASET, 'source-document')
SUSPICIOUS_FOLDER = os.path.join(DATASET, 'suspicious-document')
OUTPUT_FILE = "similarity_results.txt"

# Debugging Paths
print("Source folder:", SOURCE_FOLDER)
print("Suspicious folder:", SUSPICIOUS_FOLDER)

# Get Text Files
def get_text_files(folder):
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]

# Remove Punctuations from Text
def remove_punctuation(text):
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)

# Preprocess a Single Document
def preprocess_single_document(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            text = remove_punctuation(f.read().lower())
            return text
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return ""

# Preprocess Documents
def preprocess_documents(files):
    preprocessed_documents = []
    for i, file in enumerate(files, 1):
        text = preprocess_single_document(file)
        preprocessed_documents.append(text)
        if i % 100 == 0 or i == len(files):
            print(f"Preprocessed {i}/{len(files)} documents.")
    return preprocessed_documents

# Compute Document Frequencies and TF-IDF Vectors using TfidfVectorizer
def compute_tfidf_vectors(documents, ngram_range=(3,3)):
    print("Computing TF-IDF vectors with TfidfVectorizer...")
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    print(f"Number of unique terms (trigrams) after eliminating stopwords: {len(terms)}")
    return tfidf_matrix, terms, vectorizer

# Compute Cosine Similarity Matrix
def compute_cosine_similarities(suspicious_vectors, source_vectors):
    print("Computing cosine similarities...")
    similarity_matrix = cosine_similarity(suspicious_vectors, source_vectors)
    print("Cosine similarity computation complete.")
    return similarity_matrix

# Main Execution
if __name__ == "__main__":
    start_time = time.time()

    # Load Source and Suspicious Files
    source_files = get_text_files(SOURCE_FOLDER)
    suspicious_files = get_text_files(SUSPICIOUS_FOLDER)

    print(f"Number of source files: {len(source_files)}")
    print(f"Number of suspicious files: {len(suspicious_files)}")

    # Combine All Files for Vocabulary Creation
    all_files = source_files + suspicious_files

    # Preprocess Documents into Raw Text
    print("Preprocessing documents...")
    preprocessed_documents = preprocess_documents(all_files)

    # Compute TF-IDF Vectors using Trigrams
    tfidf_matrix, terms, vectorizer = compute_tfidf_vectors(preprocessed_documents, ngram_range=(3,3))

    # Split TF-IDF Vectors into Source and Suspicious
    source_vectors = tfidf_matrix[:len(source_files)]
    suspicious_vectors = tfidf_matrix[len(source_files):]

    # Compute Cosine Similarities
    similarity_matrix = compute_cosine_similarities(suspicious_vectors, source_vectors)

    # Write Results to File
    print(f"Writing similarity results to {OUTPUT_FILE}...")
    num_suspicious = suspicious_vectors.shape[0]  # Corrected line
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for s_idx, similarities in enumerate(similarity_matrix):
            max = 0
            maxSource = ""
            for src_idx, sim in enumerate(similarities):
                if sim > max:
                    max = sim
                    maxSource = src_idx
            out.write(f"Suspicious doc {s_idx+1}, Source doc {maxSource+1}, Similarity: {max:.4f}\n")
            # Corrected condition to use shape[0] instead of len(suspicious_vectors)
            if (s_idx + 1) % 10 == 0 or (s_idx + 1) == num_suspicious:
                print(f"Processed {s_idx + 1}/{num_suspicious} suspicious documents.")

    print(f"Processing complete. Results written to {OUTPUT_FILE}.")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

