#!/usr/bin/python3

# usage: find_frames.py hosts/hostname/system_name/

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def add_frame_numbers(df):
    """
    Add frame numbers column, where:
    - Timesteps divisible by 4000 fs get frame numbers (t_fs/4000)
    - Other timesteps get NaN
    Args:
        df: DataFrame containing 't_fs' column with timesteps in femtoseconds
    Returns:
        DataFrame with new 'frame_number' column
    """
    # Initialize frame_number column with NaN
    df['frame_number'] = pd.NA
    
    # Only assign frame numbers to timesteps divisible by 4000
    mask = (df['t_fs'] % 4000) == 0
    df.loc[mask, 'frame_number'] = df.loc[mask, 't_fs'].div(4000).astype(np.int64)
    
    # Count valid frames
    valid_frames = df['frame_number'].dropna()
    print(f"\nFrame number statistics:")
    print(f"Total timesteps: {len(df)}")
    print(f"Valid frames: {len(valid_frames)}")
    print(f"Time range: {df['t_fs'].min()} fs to {df['t_fs'].max()} fs")
    if len(valid_frames) > 0:
        print(f"Frame numbers range from {int(valid_frames.min())} to {int(valid_frames.max())}")
    else:
        print("No valid frames found!")
    
    return df

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: find_frames.py hosts/hostname/system_name/")
        sys.exit(1)
        
    full_path = sys.argv[1]
    fp = Path(full_path)
    path_parts = fp.parts
    
    if len(path_parts) < 3:
        print("error: path too short")
        sys.exit(1)
    else:
        host = path_parts[1]
        system = path_parts[2]
        
        # Read the RMSD CSV file
        dvdl_rmsd_csv_path = os.path.join(full_path, 'dvdl_rmsd.csv')
        
        if not os.path.exists(dvdl_rmsd_csv_path):
            print(f"error: data not found {dvdl_rmsd_csv_path}")
            print("Please run rmsd_values.py first to generate dvdl_rmsd.csv")
            sys.exit(2)
        
        print(f"Reading {dvdl_rmsd_csv_path}...")
        dvdl = pd.read_csv(dvdl_rmsd_csv_path)
        
        print("Adding frame numbers...")
        dvdl = add_frame_numbers(dvdl)
        
        # Convert frame_number column to Int64 type (pandas nullable integer type)
        dvdl['frame_number'] = dvdl['frame_number'].astype('Int64')
        
        # Save updated dataframe
        output_csv = os.path.join(full_path, 'dvdl_rmsd_frame.csv')
        output_parquet = os.path.join(full_path, 'dvdl_rmsd_frame.parquet')
        
        print(f"Saving to {output_csv} and {output_parquet}...")
        dvdl.to_csv(output_csv, index=False)
        dvdl.to_parquet(output_parquet, index=False)
        
        print("\nSummary of frame numbers:")
        valid_frames = dvdl['frame_number'].dropna()
        print(f"Total frames: {len(valid_frames)}")
        if len(valid_frames) > 0:
            print(f"Frame number range: {int(valid_frames.min())} to {int(valid_frames.max())}")
        print("\nFirst few rows of the dataframe:")
        print(dvdl.head()) 