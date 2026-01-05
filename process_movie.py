#!/usr/bin/env python3

# This script creates a high-quality MP4 video from .ppm frames in the specified folder.
# It uses ffmpeg with high quality settings for libopenh264 and a fallback to mpeg4 if the first attempt fails. 
# The script assumes the frames are in alphabetical order.
# Usage: ./process_movie.py /path/to/your/ppm/folder 

import os
import glob
import argparse
import subprocess

def create_movie(folder_path):
    """
    Create a 30-second MP4 video from .ppm frames in the specified folder. # Change time if needed
    The frames are assumed to be in alphabetical order.
    
    Args:
        folder_path (str): Path to folder containing .ppm frames
    """
    # Get all .ppm files in the folder and sort them
    ppm_files = sorted(glob.glob(os.path.join(folder_path, "*.ppm")))
    num_frames = len(ppm_files)
    
    if num_frames == 0:
        print(f"No .ppm files found in {folder_path}")
        return
    
    # Calculate framerate for a 30-second video # Change time if needed
    framerate = num_frames / 30.0 
    
    # Output video path
    output_path = os.path.join(folder_path, "trajectory_movie.mp4")
    
    # Use ffmpeg with high quality settings for libopenh264
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(framerate),
        '-pattern_type', 'glob',
        '-i', os.path.join(folder_path, '*.ppm'),
        '-c:v', 'libopenh264',  # Using libopenh264 which is available
        '-b:v', '8M',           # High bitrate for better quality
        '-pix_fmt', 'yuv420p',  # Required for compatibility
        output_path
    ]
    
    try:
        print(f"Creating high-quality video from {num_frames} frames...")
        subprocess.run(cmd, check=True)
        print(f"Video successfully saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        # Try a simpler fallback option if the first attempt fails
        try:
            print("Trying fallback encoding option...")
            cmd = [
                'ffmpeg',
                '-y',
                '-framerate', str(framerate),
                '-pattern_type', 'glob',
                '-i', os.path.join(folder_path, '*.ppm'),
                '-c:v', 'mpeg4',   # Fallback to mpeg4
                '-q:v', '1',       # Maximum quality for mpeg4
                '-b:v', '10M',     # Even higher bitrate for the fallback
                output_path
            ]
            subprocess.run(cmd, check=True)
            print(f"Video successfully saved to: {output_path}")
        except subprocess.CalledProcessError as e2:
            print(f"Fallback encoding also failed: {e2}")
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg to use this script.")

def main():
    parser = argparse.ArgumentParser(description='Create a high-quality MP4 video from .ppm frames')
    parser.add_argument('path', help='Path to folder containing .ppm frames')
    args = parser.parse_args()
    
    # Check if the path exists and is a directory
    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist")
        return
    
    if not os.path.isdir(args.path):
        print(f"Error: Path {args.path} is not a directory")
        return
    
    create_movie(args.path)

if __name__ == "__main__":
    main()
