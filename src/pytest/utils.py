import numpy as np
import pandas as pd

def generate_synthetic_events(num_events, sensor_size):
    """Generate synthetic event data for testing."""
    x = np.random.randint(0, sensor_size[0], num_events)
    y = np.random.randint(0, sensor_size[1], num_events)
    t = np.random.randint(0, 10000, num_events)  # Timestamps
    p = np.random.randint(0, sensor_size[2], num_events)  # Polarity
    return {'x': x, 'y': y, 't': t, 'p': p}


def read_large_file_pandas(filename, col_names=["t", "x", "y", "p"], nrows=10, chunksz= 100000):
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(filename, chunksize=chunksz,sep=' ', index_col=False, header=0, false_values=['#'], names=col_names, nrows=nrows), desc='Loading data')])
    return df