import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

df = pd.read_csv(DATA_DIR)

df.head()