#!/usr/bin/python3

# usage: process_alc.py hosts/hostname/system_name/alchemical_name

import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from geometry import *

# dvdl.dat header line
#Step              Lambda                phiX               Theta              vTheta                 Phi  ...

def read_dvdl(dvdl):
    # Only read the first two columns for efficiency
    dvdl_df = pd.read_csv(dvdl)
    print(dvdl_df.head())
    #dvdl_df = pd.read_csv(dvdl, delim_whitespace=True, skiprows=1, usecols=[0, 1, 2, 3, 4, 5]) #was to cut to 6 columns
    #dvdl_df = pd.read_csv(dvdl, delim_whitespace=True, skiprows=1, usecols=[0, 1])
    # Rename columns for clarity
    #dvdl_df.columns = ['Time (fs)', 'Lambda']
    dvdl_df.columns = ['t_fs', 'lambda', 'phi_x', 'theta', 'v_theta', 'phi',]
    # print(dvdl_df)
    return dvdl_df  

def read_fl(fl_file):
    # Only read the first two columns for efficiency
    fl = pd.read_csv(fl_file)
    # Rename columns for clarity
    fl.columns = ["𝜆", "F(𝜆) fitted", "dF/d𝜆 fitted", "F(𝜆) raw", "<dU/d𝜆> raw"]
    return fl 

def lvt_plot(df, plot_fname, system):
        # Plot the data efficiently and save to PDF
        # Create output directory if it doesn't exist
 
        # Save the figure to PDF with higher DPI for better quality
        output_dir = os.path.dirname(plot_fname)
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the efficient context manager approach for PDF creation
        with PdfPages(plot_fname) as pdf:
            # Create figure with adjusted size and padding to ensure labels are visible
            plt.figure(figsize=(10, 7))
            ax = plt.subplot(111)
            
            # Plot directly from the DataFrame - more efficient than extracting arrays
            ax.plot(df.iloc[:,0], df.iloc[:,1], 'b-', linewidth=1.5)
            #ax.plot(df['Time (fs)'], df['Lambda'], 'b-', linewidth=1.5)
            
            # Set labels and title with larger font size
            plt.xlabel('Time (fs)', fontsize=12, labelpad=10)
            plt.ylabel('λ', fontsize=12, labelpad=10)
            plt.title(f'λ vs Time for {system}', fontsize=14)
            
            # Adjust layout to make room for labels
            plt.tight_layout(pad=3.0)
            
            # Save the figure to PDF with higher DPI for better quality
            pdf.savefig(bbox_inches='tight')
            
            # Close the figure to free memory
            plt.close()
        
        print(f"Plot saved to {plot_fname}")
    
def fl_plot(df, plot_fname, system):
        # Plot the data efficiently and save to PDF
        # Create output directory if it doesn't exist
 
        # Save the figure to PDF with higher DPI for better quality
        output_dir = os.path.dirname(plot_fname)
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove consecutive points with the same dF/d𝜆 fitted value
        # First, calculate the differences between consecutive points
        df['diff'] = df["dF/d𝜆 fitted"].diff()
        # Keep points where the difference is not zero or NaN
        df = df[df['diff'].notna() & (df['diff'] != 0)]
        # Drop the temporary diff column
        df = df.drop('diff', axis=1)
        
        # Use the efficient context manager approach for PDF creation
        with PdfPages(plot_fname) as pdf:
            # Create figure with adjusted size and padding to ensure labels are visible
            plt.figure(figsize=(10, 7))
            ax = plt.subplot(111)
            
            # Plot directly from the DataFrame - more efficient than extracting arrays
            ax.plot(df.iloc[:,0], df.iloc[:,3], 'r-', linewidth=1.5)
            
            # Set labels and title with larger font size
            plt.xlabel('λ', fontsize=12, labelpad=10)
            plt.ylabel('∆G (Kcal/mol)', fontsize=12, labelpad=10)
            plt.title(f'FEL for {system}', fontsize=14)
            
            # Adjust layout to make room for labels
            plt.tight_layout(pad=3.0)
            
            # Save the figure to PDF with higher DPI for better quality
            pdf.savefig(bbox_inches='tight')
            
            # Close the figure to free memory
            plt.close()
        
        print(f"Plot saved to {plot_fname}")


if __name__ == '__main__':

    full_path = sys.argv[1]
    fp = Path(full_path)
    path_parts = fp.parts

    if len(path_parts) < 3:
        print("error: path too short")
        sys.exit(1)
    else:
        host = path_parts[1]
        system = path_parts[2]


        dvdl_csv_path = os.path.join(full_path, 'dvdl.csv')
        fl_csv_path = os.path.join(full_path, 'fl.csv')

        if not os.path.exists(dvdl_csv_path):
            print(f"error: data not found {dvdl_csv_path}")
            sys.exit(2)

        if not os.path.exists(fl_csv_path):
            print(f"error: data not found {fl_csv_path}")
            sys.exit(2)
        
        dvdl = read_dvdl(dvdl_csv_path)
        fl = read_fl(fl_csv_path)
        print(dvdl)
        print(fl)
        
        # Define output PDF path
        output_dir = "/scratch/users/lm18di/geometry/figures/"
        dvdl_pdf = os.path.join(output_dir, f"{system}_lambda_time_plot.pdf")
        fl_pdf = os.path.join(output_dir, f"{system}_fl_lambda_plot.pdf")

        lvt_plot(dvdl, dvdl_pdf, system)
        fl_plot(fl, fl_pdf, system)



        
