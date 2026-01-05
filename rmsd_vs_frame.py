import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_rmsd_vs_frame(data_path):
    # Extract system name from the path
    system = os.path.basename(data_path)
    
    # Read the CSV file
    df = pd.read_csv(os.path.join(data_path, 'dvdl_rmsd_frame.csv'))

    # Convert frame_number to numeric, coercing errors to NaN
    df['frame_number'] = pd.to_numeric(df['frame_number'], errors='coerce')

    # Drop rows where frame_number is NaN
    df = df.dropna(subset=['frame_number'])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['frame_number'], df['rmsd'], 'b-', linewidth=1)
    plt.xlabel('Frame Number')
    plt.ylabel('RMSD (Å)')
    plt.title('RMSD vs Frame Number')
    plt.grid(True)

    # Create figures directory if it doesn't exist
    figures_dir = '/scratch/users/lm18di/geometry/figures'
    os.makedirs(figures_dir, exist_ok=True)

    # Save the plot as PDF
    output_path = os.path.join(figures_dir, f'{system}_rmsd_vs_frame.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rmsd_vs_frame.py <path_to_data_directory>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    plot_rmsd_vs_frame(data_path)
