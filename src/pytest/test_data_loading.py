import numpy as np
from tqdm import tqdm
import pandas as pd

def read_large_file_pandas(filename, chunk):
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(filename, chunksize=chunk), desc='Loading data')])
    return df

# Example usage
file_path = 'datasets/uzh_fpv/Tracks/Track1/txt/events.txt'
df=read_large_file_pandas(file_path, 100000)
print(df)
