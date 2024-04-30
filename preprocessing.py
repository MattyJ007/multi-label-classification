import re
import pandas as pd
from io import StringIO
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
stop_words = set(stopwords.words('english'))

def normaliseImage(imageId):
  print(imageId)
  # Save new JPG to processed-data directory

def oneHotEncodeLabel():
   return []

def tokenise(caption):
      # Replace non-alphabetic characters with single whitespace
    caption = re.sub(r'[^a-zA-Z\s]', ' ', caption.lower())
    # Remove any whitespace that appears in sequence
    caption = re.sub(r"\s+", " ", caption)
    # Remove new leading and trailing whitespace
    caption = caption.strip()
    # Tokenize
    word_tokens = word_tokenize(caption)
    # Remove stop words
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    # Replace value with tokenised data
    return filtered_sentence

FILENAME = 'data/train.csv'

with open(FILENAME) as file:
  lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
  df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")

for row in df.itertuples():
    df.at[row.Index, 'ImageID'] = normaliseImage(row.ImageID)
    df.at[row.Index, 'Labels'] = oneHotEncodeLabel(row.Labels)
    df.at[row.Index, 'Caption'] = tokenise(row.Caption)

df.to_csv('processed-data/train.csv', index=False)
