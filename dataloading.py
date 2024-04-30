import re
import pandas as pd
from io import StringIO

FILENAME = 'train.csv'
with open(FILENAME) as file:
  lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
  df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")

print(df.head(10))

'''
1. Access subset of training data and loop through each row
2. Find largest images portrait and landscape - resize all images to same.
3. Greyscale than normalise (pixel values between 0 - 1)
4. One hot encode labels
5. Preprocess captions - remove stop words + lowercasing + tokenise
6. Visualise the data at the different stages
7. Build model with PyTorch
8. Add F1 score
9. Add improvements
 - Piggyback off of pretrained models
 - cnn layers
 - preprocessing
 - Low rank adaption + parameter efficient fine tuning
'''
