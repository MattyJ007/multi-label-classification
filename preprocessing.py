import re
import pandas as pd
from io import StringIO
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
stop_words = set(stopwords.words('english'))

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

test = df.head()
for row in test.itertuples():
  print(test.at[row.Index, 'Labels'])
  print( df["rnti"].str.split(",").apply(lambda x: [int(i, 16) for i in x]) )
# for row in df.itertuples():
#     df.at[row.Index, 'Caption'] = tokenise(row.Caption)
    

# df.to_csv('processed-data/train.csv', index=False)
