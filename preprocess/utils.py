"""
Module: utils
Contains utility functions for file operations and helper routines.
"""

import os
import shutil

def make_folders(source_path):
    """
    Create required folder structure for the project.
    
    Parameters:
        source_path (str): Base directory path.
    """
    input_p = os.path.join(source_path, 'input')
    distorted_path = os.path.join(source_path, "distorted")
    distorted_sparse_path = os.path.join(distorted_path, "sparse")
    distorted_sparse_final_path = os.path.join(distorted_path, "sparse_final")
    sparse_path = os.path.join(source_path, "sparse/0")

    os.makedirs(input_p, exist_ok=True)
    os.makedirs(distorted_path, exist_ok=True)
    os.makedirs(distorted_sparse_path, exist_ok=True)
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(distorted_sparse_final_path, exist_ok=True)

def clean_paths(source_path, video_n, db_path):
    """
    Clean the source directory by removing temporary and unwanted folders.
    
    Parameters:
        source_path (str): Base directory path.
        video_n (str): Video filename to be retained.
        db_path (str): Path to the database file to be removed.
    """
    if os.path.isfile(db_path):
        os.remove(db_path)
    all_files = os.listdir(source_path)
    if video_n in all_files:
        all_files.remove(video_n)
    if "tmp" in all_files:
        all_files.remove("tmp")
    print(f"Clean start. removing:\n{all_files}")
    paths = [os.path.join(source_path, tmp) for tmp in all_files]
    [shutil.rmtree(tmp) for tmp in paths if os.path.isdir(tmp)]

def _name_to_ind(name):
    """
    Convert an image filename to an index.
    
    Parameters:
        name (str): Image filename (e.g., "00000001.jpeg").
        
    Returns:
        int: Numeric index extracted from the filename.
    """
    ind = int(name.split('.')[0])
    return ind
