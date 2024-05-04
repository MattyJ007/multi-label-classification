import re
import pandas as pd
from io import StringIO

FILENAME = 'train.csv'
with open(FILENAME) as file:
  lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
  df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")

print(df.head(10))

'''
Done
1. Access subset of training data and loop through each row
2. Find largest images portrait and landscape - resize all images to same.
3. Greyscale than normalise (pixel values between 0 - 1) (Reshape to resnet's expected shapeg)
4. One hot encode labels + check label distribution within training set
5. Preprocess captions - remove stop words + lowercasing + tokenise

To Do
6. Visualise the data at the different stages
7. Build model with PyTorch off of ResNet18? (maybe check out other multilabel models) (motivation for resenet18 - some of our classes have very few examples - training from scratch would give very poor performance)
   - Use untrained architecture
   - use pretrained architecture
   - (optional) Map ResNet18 classes to our classes and check accuracy - play around with whether resnet18 can do multi label
   - pretrained language model
8. Save model to file
9. Add F1 score
10. Hyperparameter tuning
11. Ablation tests based on above baselines
- baseline with and without pretraining
- remove captions
- remove images
12. Add improvements
 - Piggyback off of pretrained models
 - cnn layers
 - preprocessing
 - Low rank adaption + parameter efficient fine tuning
 - Split image to increase training sets
'''
