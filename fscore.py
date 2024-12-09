import os
import sys
import xml.etree.ElementTree as ET

#fscoreArraySystem:
# Input: similarity_results.txt -- text file w/ max similarity outputted
# parse through suspicious docs, put 0 if no plagiarism found, 1 if plagiarism detectes
# Output: binary array of system, responseGroupCount

def fscoreArraySystem(systemOutputFile):
    responseGroupCount = 0
    with open(systemOutputFile, 'r') as file:
        systemOutput = file.readlines()
    
    system_array = []
    for sim in systemOutput:
        if float(sim) > 0:
            system_array.append(1)
            responseGroupCount+= 1
        else:
            system_array.append(0)
    
    return system_array, responseGroupCount
        
#fscoreKey
# Input: corpus
# Output: binary array of answer key, keyGroupCount

def fscoreKey(systemOutputFile):
    key_array = []
    """
    Counts how many suspicious documents have at least one plagiarism feature.
    
    Args:
        systemOutputFile (str): Path to the systemOutputFile containing XML files.
    
    Returns:
        keyGroupCount, total_count, key_array
    """
    keyGroupCount = 0
    total_count = 0
    # Traverse the systemOutputFile to process XML files
    for root, _, files in os.walk(systemOutputFile):
        for file in files:
            if file.endswith('.xml'):
                total_count += 1  # Count the total number of XML files
                file_path = os.path.join(root, file)
                
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

# fscoreCorrectness:
# Input: binary array of system, binary array of answer key
# Output: incorrect, correct
# Idea: compare each element in array, easy and intuitive

def fscoreCorrectness(system_array,key_array):
    correct = 0
    incorrect = 0
    correctTrue = 0
    if len(system_array) != len(key_array):
        print("Error: length of system does not match length of key")
        sys.exit()
    
    for i in range(len(system_array)):
        if system_array[i] == key_array[i] and system_array[i]==1:
            correctTrue += 1
            correct += 1
        elif system_array[i] == key_array[i]:
            correct += 1
        
    incorrect = len(system_array)- correct
    return correct, incorrect, correctTrue
    

#fscore:
# Inputs:
# - correct: number of documents correctly identified as plagiarized
# - incorrect: number of documents incorrectly identified
# - keyGroupCount: total number of plagiarism cases in the ground truth (answer key)
# - responseGroupCount: total number of plagiarism cases identified by the system (count in detector function)

def fscoreCalculate(correct, incorrect, correctTrue, keyGroupCount, responseGroupCount):

    # Print the accuracy of document-level detection
    print(f"{correct} out of {correct + incorrect} documents correctly identified.")
    accuracy = 100.0 * correct / (correct + incorrect)
    print(f"  Accuracy: {accuracy:.2f}%")

    # Print group-level detection statistics
    print(f"{keyGroupCount} groups in the answer key (ground truth).")
    print(f"{responseGroupCount} groups identified by the system (response).")
    print(f"{correct} groups correctly matched.")

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
    rounded_F1 = int(round(scaled_F1, 0))
    print(f"  Rounded Scaled F1: {rounded_F1}")

if __name__ == "__main__":
    system_array, responseGroupCount = fscoreArraySystem("similarity_results.txt")

    suspicious_docs_dir = 'training-corpus/suspicious-document'
    keyGroupCount, total_count, key_array = fscoreKey(suspicious_docs_dir)
    #print(f"Number of plagiarized documents: {plagiarized_docs}")

    correct, incorrect, correctTrue = fscoreCorrectness(system_array,key_array)
    fscoreCalculate(correct, incorrect, correctTrue, keyGroupCount, responseGroupCount)
