#!/usr/bin/python3

"""
Script to create the final summary plots that weren't generated during the main DCD analysis.
This script reuses the plotting functions from dcd_analysis.py but is optimized for memory usage.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import gc
import psutil
import time

def get_memory_usage():
    """Return the current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

def log_memory_usage(message):
    """Log memory usage with a custom message"""
    mem_usage = get_memory_usage()
    total_mem = psutil.virtual_memory().total / 1024 / 1024
    percent = mem_usage / total_mem * 100
    print(f"{message}: {mem_usage:.1f} MB ({percent:.1f}% of {total_mem:.1f} MB total)")

def plot_fl_vs_rmsd_with_minima(df, minima_info, output_path, system):
    """Create Free Energy vs RMSD plot with minima marked (memory-optimized)"""
    log_memory_usage("Before creating plot")
    
    # Ensure the dataframe only has the columns we need
    necessary_columns = ['rmsd', 'F(𝜆) raw']
    df = df[necessary_columns].copy()
    
    # Trigger garbage collection to free memory
    gc.collect()
    log_memory_usage("After preparing data")
    
    try:
        # Create the figure
        plt.figure(figsize=(10, 7))
        ax = plt.subplot(111)
        
        # Plot the main data
        ax.plot(df['rmsd'], df['F(𝜆) raw'], 'r-', linewidth=1.5)
        
        # Mark minima regions one by one to avoid high memory usage
        for i, min_range in enumerate(minima_info, 1):
            # Highlight the RMSD range region
            well_mask = (df['rmsd'] >= min_range[0]) & (df['rmsd'] <= min_range[1])
            well_points = df[well_mask]
            
            if not well_points.empty:
                print(f"Processing minima {i} - found {len(well_points)} points in range")
                rmsd_mid = (min_range[0] + min_range[1]) / 2
                
                if 'F(𝜆) raw' in well_points.columns:
                    energy_min = well_points['F(𝜆) raw'].min()
                    
                    # Fill between
                    well_rmsd = well_points['rmsd'].values
                    well_energy = well_points['F(𝜆) raw'].values
                    energy_max = well_points['F(𝜆) raw'].max()
                    
                    ax.fill_between(well_rmsd, well_energy, [energy_max] * len(well_rmsd), 
                                  color='red', alpha=0.2)
                    
                    ax.scatter(rmsd_mid, energy_min, color='blue', s=100, zorder=5)
                    ax.annotate(f'Min {i}', 
                              (rmsd_mid, energy_min),
                              xytext=(10, 10), textcoords='offset points',
                              fontsize=10, color='blue')
                else:
                    print(f"Warning: 'F(𝜆) raw' column not found in well_points for minima {i}")
            
            # Free memory between iterations
            del well_points
            gc.collect()
        
        plt.xlabel('RMSD (Å)', fontsize=12, labelpad=10)
        plt.ylabel('∆G (Kcal/mol)', fontsize=12, labelpad=10)
        plt.title(f'FEL vs RMSD for {system} (with Minima Regions)', fontsize=14)
        
        plt.tight_layout(pad=3.0)
        
        log_memory_usage("Before saving plot")
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        gc.collect()
        
        print(f"Plot saved to {output_path}")
        log_memory_usage("After saving plot")
        
        return True
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        # Ensure plot is closed even if there's an error
        try:
            plt.close()
        except:
            pass
        gc.collect()
        return False

def main():
    # Track start time
    start_time = time.time()
    
    # Print system info
    print(f"Python version: {sys.version}")
    print(f"Memory info: {psutil.virtual_memory()}")
    log_memory_usage("At start")
    
    # Define the system we're working with
    system = "run_cam_c28w"
    
    # Define paths
    root_path = "/scratch/users/lm18di/geometry"
    data_dir = os.path.join(root_path, "hosts/static", system)
    output_dir = os.path.join(root_path, "figures")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths for plots
    fl_plot = os.path.join(output_dir, f"{system}_fl_with_minima.pdf")
    
    # Define the RMSD ranges for minima (same as in dcd_analysis.py)
    minima_ranges = [
        (4.417, 4.551),    # Minimum 1
        (5.422, 5.623),    # Minimum 2
        (6.3265, 6.494),   # Minimum 3
        (7.834, 8.0685),   # Minimum 4
        (10.447, 10.648)    # Minimum 5
    ]
    
    # Load data files
    fl_rmsd_csv_path = os.path.join(data_dir, 'fl_rmsd.csv')
    
    if not os.path.exists(fl_rmsd_csv_path):
        print(f"Error: File not found: {fl_rmsd_csv_path}")
        sys.exit(1)
    
    try:
        print(f"Loading data file: {fl_rmsd_csv_path}")
        fl_df = pd.read_csv(fl_rmsd_csv_path)
        print(f"Loaded {len(fl_df)} rows from fl_rmsd.csv")
        log_memory_usage("After loading data")
        
        # Print dataframe info
        print("\nDataframe information:")
        print(f"Columns: {fl_df.columns.tolist()}")
        print(f"Data types: {fl_df.dtypes}")
        print(f"Memory usage: {fl_df.memory_usage().sum() / 1024 / 1024:.2f} MB")
        
        # Create the plot
        print(f"\nCreating FEL vs RMSD plot...")
        success = plot_fl_vs_rmsd_with_minima(fl_df, minima_ranges, fl_plot, system)
        
        if success:
            print("\nAnalysis completed successfully")
        else:
            print("\nFailed to create plot")
        
        # Clean up for good measure
        del fl_df
        gc.collect()
        
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.1f} seconds")
        log_memory_usage("Final memory usage")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 