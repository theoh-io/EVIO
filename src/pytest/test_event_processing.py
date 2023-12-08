# Plot the optical flow based on the SAEs faster than before. Corrected and improved. 
# 2 methods for plane fitting (the second one doesn't work properly)
# The summations to compute of terms needed to obtain full flow are implemented with numpy.sum() -> extremely faster than for loops
# Normal flow and sparse or dense full flow computed.
  
import cv2 # OpenCV library
#----------------------------------------------
import math
from scipy.linalg import svd
from time import time
import colorsys
#----------------------------------------------
import numpy as np
from src.options.config_parser import ConfigParser
from src.representations.processor import EventProcessor
from tqdm import tqdm
import pandas as pd

PI = 3.1415927


COLS=346
ROWS=260



CHAN = 2
SAESIZE = 5
APERTURE_THLD_SAE = 10**(9) # Ratio between evals in SAE problem, the solutions must be well conditioned (rank 2) to have sufficient edge support in both plane directions
MINHITS = 5 # Minimum number of hits in SAE to track an edge
DELTA_TIME = 30**(-2) # Maximum allowed time within SAE in seconds (10ms)
MUCHOS = 300000 # An event arrives each microsec during 30ms, maximum num of events in a message

# MAXV_FF = 10 # Maximum module velocity so the image isn't white (normalization of the vel module for the representation), full flow
# MAXV_NF = 20 # Maximum module velocity so the image isn't white (normalization of the vel module for the representation), normal flow

# FF_WINDOW_SIZE = 35 # Full flow window size
# HALF_FF_WINDOW_SIZE = int(FF_WINDOW_SIZE/2)

# PLOT_NF = 1 # 1 if plot normal flow
# PLOT_FF = 1 # 1 if plot full flow

# PLANE_FITTING_METHOD = 1 # 1 for method 1 (SVD), 2 for method 2 (simplest way)

# FULL_FLOW = 1 # 1 to compute full flow
# SPARSE = 1 # Compute sparse full flow (not 1 when DENSE = 1)
# DENSE = 0 # Compute dense full flow (not 1 when SPARSE = 1)








def read_large_file_pandas(filename, nrows=10, chunksz= 100000):
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(filename, chunksize=chunksz, nrows=nrows), desc='Loading data')])
    return df



def main(args=None):
  
  # Load data from txt

  # Example usage
  file_path = 'datasets/uzh_fpv/Tracks/Track1/txt/events.txt'
  df=read_large_file_pandas(file_path, nrows=50, chunksz = 100000)
 
  event_batch=df[:10]
  print(event_batch)
  
  method="sae" #SAE
  ev_processor=EventProcessor(method=method)

  result=ev_processor.process_batch(event_batch)
  print(f"result: {result}")

  
if __name__ == '__main__':
  main()