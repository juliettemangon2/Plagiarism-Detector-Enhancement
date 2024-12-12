import os
import sys
import numpy as np
from string import punctuation
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import argparse  # New: Import argparse for command-line argument parsing
from concurrent.futures import ProcessPoolExecutor

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')

# ---------------------------- Configuration Constants ---------------------------- #

MEASURE = 'cosine'  # Options: 'cosine', 'jaccard'
DATASET = 'training-corpus'  # Name of your corpus
SOURCE_FOLDER = os.path.join(DATASET, 'source-document')
SUSPICIOUS_FOLDER = os.path.join(DATASET, 'suspicious-document')
OUTPUT_FILE_TEMPLATE = "similarity_results_ngrams_{ng}_thresh_{thresh}.txt"

# Define ranges for n-grams and similarity thresholds
NGRAM_MIN = 1
NGRAM_MAX = 15  # Adjust as needed (Be cautious with high values due to computational constraints)
THRESH_MIN = 0.0000
THRESH_MAX = 0.0100
THRESH_STEP = 0.0002

# ----------------------------------------------------------------------------------- #

# ----------------------------- Helper Functions ------------------------------------ #

def get_text_files(folder):
    """
    Retrieves all .txt files from the specified folder.

    Args:
        folder (str): Path to the folder containing text files.

    Returns:
        list: List of file paths ending with .txt.
    """
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]

def remove_punctuation(text):
    """
    Removes punctuation from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without punctuation.
    """
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)

def preprocess_single_document(file):
    """
    Reads and preprocesses a single document.

    Args:
        file (str): Path to the text file.

    Returns:
        str: Preprocessed text.
    """
    try:
        with open(file, 'r', encoding='utf-8') as f:
            text = remove_punctuation(f.read().lower())
            return text
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return ""

def preprocess_documents(files):
    """
    Preprocesses a list of documents.

    Args:
        files (list): List of file paths.

    Returns:
        list: List of preprocessed texts.
    """
    preprocessed_documents = []
    for i, file in enumerate(files, 1):
        text = preprocess_single_document(file)
        preprocessed_documents.append(text)
        if i % 100 == 0 or i == len(files):
            print(f"Preprocessed {i}/{len(files)} documents.")
    return preprocessed_documents

def compute_tfidf_vectors(documents, ngram_range=(3,3)):
    """
    Computes TF-IDF vectors for the given documents.

    Args:
        documents (list): List of preprocessed texts.
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams.

    Returns:
        tuple: TF-IDF matrix, list of terms, and the vectorizer object.
    """
    print("Computing TF-IDF vectors with TfidfVectorizer...")
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    print(f"Number of unique terms (n-grams) after eliminating stopwords: {len(terms)}")
    return tfidf_matrix, terms, vectorizer

def compute_cosine_similarities(suspicious_vectors, source_vectors):
    """
    Computes cosine similarity between suspicious and source documents.

    Args:
        suspicious_vectors (csr_matrix): TF-IDF vectors of suspicious documents.
        source_vectors (csr_matrix): TF-IDF vectors of source documents.

    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    print("Computing cosine similarities...")
    similarity_matrix = cosine_similarity(suspicious_vectors, source_vectors)
    print("Cosine similarity computation complete.")
    return similarity_matrix

def compute_jaccard_similarities(suspicious_vectors, source_vectors):
    """
    Computes Jaccard similarity between suspicious and source documents.

    Args:
        suspicious_vectors (csr_matrix): TF-IDF vectors of suspicious documents.
        source_vectors (csr_matrix): TF-IDF vectors of source documents.

    Returns:
        np.ndarray: Jaccard similarity matrix.
    """
    print("Computing Jaccard similarities...")
    # Binarize the TF-IDF matrices
    suspicious_binary = (suspicious_vectors > 0).astype(int)
    source_binary = (source_vectors > 0).astype(int)
    
    # Convert to CSR format for efficient row operations
    suspicious_binary = suspicious_binary.tocsr()
    source_binary = source_binary.tocsr()
    
    # Compute the intersection (A AND B)
    intersection = suspicious_binary.dot(source_binary.T).toarray()
    
    # Compute the number of non-zero elements in each document (A OR B = A + B - A AND B)
    suspicious_counts = suspicious_binary.getnnz(axis=1).reshape(-1,1)
    source_counts = source_binary.getnnz(axis=1).reshape(1,-1)
    
    # Compute the union
    union = suspicious_counts + source_counts - intersection
    
    # Avoid division by zero
    union[union == 0] = 1e-10
    
    # Compute Jaccard similarity
    similarity_matrix = intersection / union
    
    print("Jaccard similarity computation complete.")
    return similarity_matrix

def fscoreArraySystem(similarity_scores, threshold):
    """
    Converts similarity scores to binary array based on threshold.

    Args:
        similarity_scores (list of float): List of similarity scores.
        threshold (float): Threshold to determine plagiarism detection.

    Returns:
        tuple: Binary array and count of detections.
    """
    responseGroupCount = 0
    system_array = []
    for sim in similarity_scores:
        if sim >= threshold:
            system_array.append(1)
            responseGroupCount += 1
        else:
            system_array.append(0)
    return system_array, responseGroupCount

def fscoreKey(suspicious_docs_dir):
    """
    Generates the answer key based on XML files indicating plagiarism.

    Args:
        suspicious_docs_dir (str): Path to the directory containing suspicious documents with XML files.

    Returns:
        tuple: Number of plagiarism cases in ground truth, total number of suspicious documents, binary array.
    """
    key_array = []
    keyGroupCount = 0
    total_count = 0
    # Traverse the suspicious_docs_dir to process XML files
    for root_dir, _, files in os.walk(suspicious_docs_dir):
        for file in files:
            if file.endswith('.xml'):
                total_count += 1  # Count the total number of XML files
                file_path = os.path.join(root_dir, file)
                
                # Parse the XML file
                try:
                    tree = ET.parse(file_path)
                    root_element = tree.getroot()
                    # Check for any 'plagiarism' feature in the XML
                    plagiarism_features = root_element.findall(".//feature[@name='plagiarism']")
                    if plagiarism_features:
                        keyGroupCount += 1
                        key_array.append(1)
                    else:
                        key_array.append(0)
                except ET.ParseError:
                    print(f"Error parsing XML file: {file_path}")
    return keyGroupCount, total_count, key_array

def fscoreCorrectness(system_array, key_array):
    """
    Compares system detections with ground truth to calculate correctness.

    Args:
        system_array (list of int): Binary array from the system.
        key_array (list of int): Binary array from the ground truth.

    Returns:
        tuple: Number of correct detections, incorrect detections, and true positives.
    """
    correct = 0
    incorrect = 0
    correctTrue = 0
    if len(system_array) != len(key_array):
        print("Error: Length of system array does not match length of key array.")
        sys.exit()
    
    for i in range(len(system_array)):
        if system_array[i] == key_array[i]:
            correct += 1
            if system_array[i] == 1:
                correctTrue += 1
        else:
            incorrect += 1
    
    return correct, incorrect, correctTrue

def fscoreCalculate(correct, incorrect, correctTrue, keyGroupCount, responseGroupCount):
    """
    Calculates and prints precision, recall, and F1-score.

    Args:
        correct (int): Number of correct detections.
        incorrect (int): Number of incorrect detections.
        correctTrue (int): Number of true positive detections.
        keyGroupCount (int): Total number of plagiarism cases in ground truth.
        responseGroupCount (int): Total number of plagiarism cases detected by the system.

    Returns:
        float: Scaled F1-score.
    """
    # Print the accuracy of document-level detection
    print(f"{correct} out of {correct + incorrect} documents correctly identified.")
    accuracy = 100.0 * correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
    print(f"  Accuracy: {accuracy:.2f}%")

    # Print group-level detection statistics
    print(f"{keyGroupCount} groups in the answer key (ground truth).")
    print(f"{responseGroupCount} groups identified by the system (response).")
    print(f"{correctTrue} groups correctly matched.")

    # Calculate precision, recall, and F1 score
    precision = 100.0 * (correctTrue / responseGroupCount) if responseGroupCount > 0 else 0.0
    recall = 100.0 * (correctTrue / keyGroupCount) if keyGroupCount > 0 else 0.0
    F1 = ((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Scale F1 to highlight its significance (optional adjustment)
    scaled_F1 = F1 * 0.10

    # Print the detailed metrics
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1 Score:  {F1:.2f}")
    print(f"  Scaled F1: {scaled_F1:.2f}")

    # Round the scaled F1 score for reporting purposes
    rounded_F1 = round(scaled_F1, 2)
    return rounded_F1

def plot_fscore_heatmap(results_df, measure, corpus_name):
    """
    Plots a heatmap of F1-scores based on n-grams and similarity thresholds.

    Args:
        results_df (DataFrame): Pandas DataFrame containing F1-scores with n-grams as rows and thresholds as columns.
        measure (str): Similarity measure used ('cosine' or 'jaccard').
        corpus_name (str): Name of the corpus/dataset.
    """
    # Set the font to Python's default sans-serif
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # DejaVu Sans is Matplotlib's default sans-serif font

    plt.figure(figsize=(24, 18))  # Increased figure size for better readability

    sns.heatmap(
        results_df,
        cmap='viridis',
        annot=False,  # Disable annotations for a cleaner look
        fmt=".2f",
        cbar_kws={'label': 'Scaled F1-Score'}
    )

    # Adjust the title position using the 'y' parameter and include corpus name
    plt.title(
        f'Heatmap of Scaled F1-Score vs. n-grams and Similarity Threshold ({measure.capitalize()})\nCorpus: {corpus_name}',
        fontsize=20,
        y=1.05  # Lower the title slightly
    )

    plt.xlabel('Similarity Threshold', fontsize=16)
    plt.ylabel('n-grams', fontsize=16)

    # Customize X-axis tick labels to display every other threshold
    threshold_labels = [f"{thresh:.4f}" for thresh in results_df.columns]

    # Determine the step for label display (e.g., every 2nd label)
    step = 2  # Change to 3 or 4 if labels are still too dense

    # Generate tick positions centered on each heatmap cell
    tick_positions = np.arange(0.5, len(threshold_labels), step)

    # Select labels to display
    selected_labels = [threshold_labels[i] for i in range(0, len(threshold_labels), step)]

    plt.xticks(
        ticks=tick_positions,
        labels=selected_labels,
        rotation=45,
        fontsize=12
    )

    # Adjust Y-axis tick labels font size
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()
    file_name = 'fscore_heatmap.png'
    plt.savefig(file_name)
    print(f"\nHeatmap saved as '{file_name}'.")

def plot_heatmap_from_csv(csv_path, measure, corpus_name):
    """
    Reads the fscore_results.csv file and plots the heatmap.

    Args:
        csv_path (str): Path to the fscore_results.csv file.
        measure (str): Similarity measure used ('cosine' or 'jaccard').
        corpus_name (str): Name of the corpus/dataset.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit()
    
    print(f"Reading CSV file: {csv_path}")
    fscore_results_df = pd.read_csv(csv_path)
    
    # Pivot the DataFrame for Heatmap
    pivot_df = fscore_results_df.pivot(index='n-grams', columns='similarity_threshold', values='Scaled_F1')
    
    # Plot Heatmap
    plot_fscore_heatmap(pivot_df, measure, corpus_name)

def process_ngram_range(ng, preprocessed_documents, source_files, key_array, keyGroupCount):
    ngram_range = (ng, ng)
    print(f"\nProcessing n-grams: {ng}")

    # Compute TF-IDF Vectors using N-grams
    tfidf_matrix, terms, vectorizer = compute_tfidf_vectors(preprocessed_documents, ngram_range=ngram_range)

    # Split TF-IDF Vectors into Source and Suspicious
    source_vectors = tfidf_matrix[:len(source_files)]
    suspicious_vectors = tfidf_matrix[len(source_files):]

    results = []

    # Iterate over similarity thresholds
    for thresh in np.arange(THRESH_MIN, THRESH_MAX + THRESH_STEP, THRESH_STEP):
        thresh = round(thresh, 4)
        print(f"  Applying similarity threshold: {thresh}")

        # Compute Similarity Matrix
        if MEASURE.lower() == 'cosine':
            similarity_matrix = compute_cosine_similarities(suspicious_vectors, source_vectors)
        elif MEASURE.lower() == 'jaccard':
            similarity_matrix = compute_jaccard_similarities(suspicious_vectors, source_vectors)
        else:
            print(f"Unsupported MEASURE: {MEASURE}. Supported measures are 'cosine' and 'jaccard'. Exiting.")
            sys.exit()

        # Extract the maximum similarity per suspicious document
        similarity_scores = similarity_matrix.max(axis=1).tolist()

        # Apply threshold to get system array
        system_array, responseGroupCount = fscoreArraySystem(similarity_scores, thresh)

        # Compute correctness metrics
        correct, incorrect, correctTrue = fscoreCorrectness(system_array, key_array)

        # Calculate F1-score
        scaled_F1 = fscoreCalculate(correct, incorrect, correctTrue, keyGroupCount, responseGroupCount)

        # Store the result as a dictionary
        result = {
            'n-grams': ng,
            'similarity_threshold': thresh,
            'Scaled_F1': scaled_F1
        }
        results.append(result)

    return results
# ----------------------------------------------------------------------------------- #

# ----------------------------------- Main Execution --------------------------------- #

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Plagiarism Detection and Heatmap Generation")
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help="Path to an existing fscore_results.csv file to generate a heatmap."
    )
    args = parser.parse_args()

    start_time = time.time()

    if args.csv:
        # Mode 2: Generate heatmap from existing CSV
        plot_heatmap_from_csv(args.csv, MEASURE, DATASET)
    else:
        # Mode 1: Process dataset and generate heatmap
        # Load Source and Suspicious Files
        source_files = get_text_files(SOURCE_FOLDER)
        suspicious_files = get_text_files(SUSPICIOUS_FOLDER)

        print(f"Number of source files: {len(source_files)}")
        print(f"Number of suspicious files: {len(suspicious_files)}")

        if not source_files or not suspicious_files:
            print("Insufficient files to process. Exiting.")
            sys.exit()

        # Combine All Files for Vocabulary Creation
        all_files = source_files + suspicious_files

        # Preprocess Documents into Raw Text
        print("Preprocessing documents...")
        preprocessed_documents = preprocess_documents(all_files)

        # Load Ground Truth (Answer Key)
        print("Loading ground truth from XML files...")
        keyGroupCount, total_count, key_array = fscoreKey(SUSPICIOUS_FOLDER)
        print(f"Total suspicious documents (XML files): {total_count}")
        print(f"Plagiarized documents in ground truth: {keyGroupCount}")

        # Initialize list to store results
        fscore_results = []

        # Use ProcessPoolExecutor for multiprocessing
        with ProcessPoolExecutor() as executor:
            # Submit tasks for each n-gram range
            futures = [
                executor.submit(
                    process_ngram_range, 
                    ng, 
                    preprocessed_documents, 
                    source_files, 
                    key_array, 
                    keyGroupCount
                )
                for ng in range(NGRAM_MIN, NGRAM_MAX + 1)
            ]

            # Collect results as they complete
            for future in futures:
                try:
                    fscore_results.extend(future.result())
                except Exception as e:
                    print(f"Error occurred: {e}")

        # Convert the list of dictionaries to a DataFrame
        fscore_results_df = pd.DataFrame(fscore_results)

        # Pivot the DataFrame for Heatmap
        pivot_df = fscore_results_df.pivot(index='n-grams', columns='similarity_threshold', values='Scaled_F1')



    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    # Plot Heatmap
    plot_fscore_heatmap(pivot_df, MEASURE, DATASET)

    # Optionally, save the results to a CSV file
    fscore_results_df.to_csv('fscore_results.csv', index=False)
    print("\nF1-score results saved to 'fscore_results.csv'.")
