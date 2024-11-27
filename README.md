main.py README

How to run:
1) training set structure should look like: <br />
   <br />
external-detection-corpus-training/ <br />
├── source-document/ <br />
│   ├── part1/ <br />
│       ├── source-documentXXXXXX.txt (many more) <br />
├── suspicious-document/ <br />
│   ├── part1/ <br />
│       ├── suspicious-documentXXXXXX.txt (many more) <br />
<br />
2) change params <br />
MODEL = 'trigram'  # Choose 'unigram', 'bigram', or 'trigram' <br />
MEASURE = 'cosine'  # Choose 'cosine' or 'jaccard' <br />
DATASET = '/Users/walkertupman/Downloads/external-detection-corpus-training' <br />
SOURCE_FOLDER = os.path.join(DATASET, 'source-document/part1') <br />
SUSPICIOUS_FOLDER = os.path.join(DATASET, 'suspicious-document/part1') <br />
<br />
^ example <br />
<br />
3) run main.py <br />
