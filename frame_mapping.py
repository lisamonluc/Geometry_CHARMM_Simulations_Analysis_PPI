# Usage: python frame_mapping.py <original_dcd> <sampled_dcd> <sampled_frame_number>
# Example: python frame_mapping.py original.dcd sampled.dcd 10

import os
import sys
import numpy as np
import pandas as pd
import MDAnalysis as mda
from pathlib import Path

class FrameMapper:
    def __init__(self, original_dcd, sampled_dcd, mapping_file=None):
        """
        Initialize the frame mapper with original and sampled DCD files.
        
        Parameters:
        -----------
        original_dcd : str
            Path to the original DCD file
        sampled_dcd : str
            Path to the sampled DCD file
        mapping_file : str, optional
            Path to save/load the frame mapping
        """
        self.original_dcd = original_dcd
        self.sampled_dcd = sampled_dcd
        self.mapping_file = mapping_file
        
        # Load the trajectories
        self.original_u = mda.Universe(original_dcd)
        self.sampled_u = mda.Universe(sampled_dcd)
        
        # Initialize mapping
        self.original_to_sampled = {}
        self.sampled_to_original = {}
        
        # If mapping file exists, load it
        if mapping_file and os.path.exists(mapping_file):
            self.load_mapping()
        else:
            self.create_mapping()
            if mapping_file:
                self.save_mapping()
    
    def create_mapping(self):
        """Create mapping between original and sampled frames"""
        print("Creating frame mapping...")
        
        # Get total frames
        original_frames = len(self.original_u.trajectory)
        sampled_frames = len(self.sampled_u.trajectory)
        
        # Create mapping arrays
        self.original_to_sampled = np.full(original_frames, -1, dtype=int)
        self.sampled_to_original = np.full(sampled_frames, -1, dtype=int)
        
        # Compare frames to find matches
        for sampled_frame in range(sampled_frames):
            self.sampled_u.trajectory[sampled_frame]
            sampled_positions = self.sampled_u.atoms.positions
            
            # Compare with each original frame
            for original_frame in range(original_frames):
                self.original_u.trajectory[original_frame]
                original_positions = self.original_u.atoms.positions
                
                # If positions match, we found a mapping
                if np.allclose(sampled_positions, original_positions, atol=1e-6):
                    self.original_to_sampled[original_frame] = sampled_frame
                    self.sampled_to_original[sampled_frame] = original_frame
                    break
        
        print(f"Mapping created: {np.sum(self.original_to_sampled != -1)} frames mapped")
    
    def save_mapping(self):
        """Save the frame mapping to a file"""
        if not self.mapping_file:
            return
        
        # Create DataFrame with mappings
        df = pd.DataFrame({
            'original_frame': np.where(self.original_to_sampled != -1)[0],
            'sampled_frame': self.original_to_sampled[self.original_to_sampled != -1]
        })
        
        # Save to CSV
        df.to_csv(self.mapping_file, index=False)
        print(f"Frame mapping saved to {self.mapping_file}")
    
    def load_mapping(self):
        """Load frame mapping from file"""
        if not self.mapping_file or not os.path.exists(self.mapping_file):
            return
        
        # Load DataFrame
        df = pd.read_csv(self.mapping_file)
        
        # Create mapping arrays
        original_frames = len(self.original_u.trajectory)
        sampled_frames = len(self.sampled_u.trajectory)
        
        self.original_to_sampled = np.full(original_frames, -1, dtype=int)
        self.sampled_to_original = np.full(sampled_frames, -1, dtype=int)
        
        # Fill mappings
        for _, row in df.iterrows():
            self.original_to_sampled[row['original_frame']] = row['sampled_frame']
            self.sampled_to_original[row['sampled_frame']] = row['original_frame']
        
        print(f"Frame mapping loaded from {self.mapping_file}")
    
    def get_original_frame(self, sampled_frame):
        """Get the original frame number for a sampled frame"""
        if sampled_frame < 0 or sampled_frame >= len(self.sampled_to_original):
            raise ValueError(f"Invalid sampled frame number: {sampled_frame}")
        
        original_frame = self.sampled_to_original[sampled_frame]
        if original_frame == -1:
            raise ValueError(f"No mapping found for sampled frame {sampled_frame}")
        
        return original_frame
    
    def get_sampled_frame(self, original_frame):
        """Get the sampled frame number for an original frame"""
        if original_frame < 0 or original_frame >= len(self.original_to_sampled):
            raise ValueError(f"Invalid original frame number: {original_frame}")
        
        sampled_frame = self.original_to_sampled[original_frame]
        if sampled_frame == -1:
            raise ValueError(f"No mapping found for original frame {original_frame}")
        
        return sampled_frame

def find_original_frame(original_dcd, sampled_dcd, sampled_frame_number):
    """
    Find the corresponding frame in the original DCD file for a given frame in the sampled DCD file.
    
    Parameters:
    -----------
    original_dcd : str
        Path to the original DCD file
    sampled_dcd : str
        Path to the sampled DCD file
    sampled_frame_number : int
        Frame number in the sampled DCD file to find in the original DCD file
    
    Returns:
    --------
    int
        The corresponding frame number in the original DCD file
    """
    # Load the trajectories
    print(f"Loading trajectories...")
    original_u = mda.Universe(original_dcd)
    sampled_u = mda.Universe(sampled_dcd)
    
    # Check if sampled frame number is valid
    if sampled_frame_number >= len(sampled_u.trajectory):
        raise ValueError(f"Sampled frame number {sampled_frame_number} is out of range. "
                        f"Sampled trajectory has {len(sampled_u.trajectory)} frames.")
    
    # Get the positions from the sampled frame
    print(f"Getting positions from sampled frame {sampled_frame_number}...")
    sampled_u.trajectory[sampled_frame_number]
    sampled_positions = sampled_u.atoms.positions
    
    # Compare with each original frame
    print(f"Searching for matching frame in original trajectory...")
    total_frames = len(original_u.trajectory)
    
    for original_frame in range(total_frames):
        if original_frame % 1000 == 0:  # Progress update
            print(f"Checked {original_frame}/{total_frames} frames...")
        
        original_u.trajectory[original_frame]
        original_positions = original_u.atoms.positions
        
        # If positions match, we found our frame
        if np.allclose(sampled_positions, original_positions, atol=1e-6):
            print(f"\nFound match! Sampled frame {sampled_frame_number} corresponds to original frame {original_frame}")
            return original_frame
    
    raise ValueError(f"No matching frame found in original trajectory for sampled frame {sampled_frame_number}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python frame_mapping.py <original_dcd> <sampled_dcd> <sampled_frame_number>")
        sys.exit(1)
    
    original_dcd = sys.argv[1]
    sampled_dcd = sys.argv[2]
    sampled_frame_number = int(sys.argv[3])
    
    try:
        original_frame = find_original_frame(original_dcd, sampled_dcd, sampled_frame_number)
        print(f"Original frame number: {original_frame}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 