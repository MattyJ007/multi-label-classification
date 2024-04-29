import re
import pandas as pd
from io import StringIO

FILENAME = 'train.csv'
with open(FILENAME) as file:
  lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")

print(df)