import os
import xml.etree.ElementTree as ET
import shutil

source_dir = 'pan-plagiarism-corpus-2011/New_new_new_dataset_12.9.2024/sus_docs'
destination_dir = 'pan-plagiarism-corpus-2011/New_new_new_dataset_12.9.2024/source_docs'
search_dir = 'pan-plagiarism-corpus-2011/external-detection-corpus/source-document'

for filename in os.listdir(source_dir):
    if filename.endswith('.xml'):
        file_path = os.path.join(source_dir, filename)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            plagiarism_features = root.findall(".//feature[@name='plagiarism']")
            
            for feature in plagiarism_features:
                source_reference = feature.get('source_reference')
                
                if source_reference:
                    source_txt_path = os.path.join(search_dir, source_reference)
                    if os.path.exists(source_txt_path):
                        dest_folder = os.path.join(destination_dir, source_reference[:-4])
                        os.makedirs(dest_folder, exist_ok=True)
                        
                        # Copy the XML file
                        shutil.copy2(file_path, os.path.join(dest_folder, filename))
                        
                        # Copy the source TXT file
                        shutil.copy2(source_txt_path, dest_folder)
                        
                        print(f"Copied {filename} and {source_reference} to {dest_folder}")
                        break
                    else:
                        print(f"Source file {source_reference} not found for {filename}")
                else:
                    print(f"No 'source_reference' found in {filename}")
        
        except ET.ParseError:
            print(f"Error: Could not parse {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
