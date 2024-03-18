from scipy.spatial.transform import Rotation as R
import numpy as np
from tqdm import tqdm
import pandas as pd

def read_large_file_pandas(filename, chunk):
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(filename, chunksize=chunk), desc='Loading data')])
    return df

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.
    
    Parameters:
    roll (float): Rotation angle around the x-axis in radians.
    pitch (float): Rotation angle around the y-axis in radians.
    yaw (float): Rotation angle around the z-axis in radians.

    Returns:
    quaternion (numpy.ndarray): The corresponding quaternion [x, y, z, w].
    """
    # Create a Rotation object from Euler angles and convert to quaternion
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    quaternion = rotation.as_quat()  # Returns [x, y, z, w]
    return quaternion

def quaternion_to_euler(quaternion):
    rotation = R.from_quat(quaternion)
    euler = R.as_euler(rotation)
    return euler


# Read the ground truth file

file_path = 'datasets/uzh_fpv/Tracks/Track1/txt/groundtruth.txt'
df=read_large_file_pandas(file_path, 100000)
print(df)

# Compute the relative translation and rotation between timesteps
max=10
start=1200

for i in range(max):
    gt=df[i]
    print(gt)
    quat=gt[4:]
    euler=quaternion_to_euler(quat)
    print(euler)
    
# Compute the total displacement check that it matches the sum of all individual displacements.
