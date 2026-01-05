#!/usr/bin/python3

# usage: rmsd_values.py hosts/hostname/system_name/

import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from geometry import *

# cam_c28w: slope = 3.35, intercept = 1.0
# bn_bs: slope = 11.5, intercept = 1.0

def add_rmsd_values(df, lambda_col, rmsd_col, slope=3.35, intercept=1.0): # Change slope depending on the system, minc value is when lambda = 0, maxc value is when lambda = 1
    """Add RMSD values to dataframe based on lambda values using the equation RMSD = slope * lambda + intercept"""
    df[rmsd_col] = slope * df[lambda_col] + intercept
    return df

def rmsd_lvt_plot(df, plot_fname, system):
    """Plot RMSD vs Time"""
    output_dir = os.path.dirname(plot_fname)
    os.makedirs(output_dir, exist_ok=True)
    
    with PdfPages(plot_fname) as pdf:
        plt.figure(figsize=(10, 7))
        ax = plt.subplot(111)
        
        ax.plot(df['t_fs'], df['rmsd'], 'b-', linewidth=1.5)
        
        plt.xlabel('Time (fs)', fontsize=12, labelpad=10)
        plt.ylabel('RMSD (Å)', fontsize=12, labelpad=10)
        plt.title(f'RMSD vs Time for {system}', fontsize=14)
        
        plt.tight_layout(pad=3.0)
        pdf.savefig(bbox_inches='tight')
        plt.close()
    
    print(f"Plot saved to {plot_fname}")

def rmsd_fl_plot(df, plot_fname, system):
    """Plot Free Energy vs RMSD"""
    output_dir = os.path.dirname(plot_fname)
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove consecutive points with the same dF/d𝜆 fitted value
    df['diff'] = df["dF/d𝜆 fitted"].diff()
    df = df[df['diff'].notna() & (df['diff'] != 0)]
    df = df.drop('diff', axis=1)
    
    with PdfPages(plot_fname) as pdf:
        plt.figure(figsize=(10, 7))
        ax = plt.subplot(111)
        
        ax.plot(df['rmsd'], df['F(𝜆) raw'], 'r-', linewidth=1.5)
        
        plt.xlabel('RMSD (Å)', fontsize=12, labelpad=10)
        plt.ylabel('∆G (Kcal/mol)', fontsize=12, labelpad=10)
        plt.title(f'FEL for {system}', fontsize=14)
        
        plt.tight_layout(pad=3.0)
        pdf.savefig(bbox_inches='tight')
        plt.close()
    
    print(f"Plot saved to {plot_fname}")

def create_interactive_plot(df, output_path):
    """Create an interactive Plotly plot for analyzing minima"""
    fig = go.Figure()
    
    # Add the main energy landscape trace
    fig.add_trace(go.Scatter(
        x=df['rmsd'],
        y=df['F(𝜆) raw'],
        mode='lines',
        name='Free Energy',
        line=dict(color='blue', width=2)
    ))
    
    # Add hover information
    fig.update_traces(
        hovertemplate="<br>".join([
            "RMSD: %{x:.2f} Å",
            "Energy: %{y:.2f} kcal/mol"
        ])
    )
    
    # Update layout
    fig.update_layout(
        title='Interactive Free Energy Landscape',
        xaxis_title='RMSD (Å)',
        yaxis_title='Free Energy (kcal/mol)',
        hovermode='x unified',
        showlegend=True,
        height=800,
        width=1200,
        template='plotly_white'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Save the plot
    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")
    
    # Print instructions for the user
    print("\nInstructions for analyzing minima:")
    print("1. Use the interactive plot to identify minima locations")
    print("2. For each minimum, note:")
    print("   - Left and right basin boundaries (where you want to collect the frames)")
    print("3. Use this information to create a minima dictionary for dcd_analysis.py")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: rmsd_values.py hosts/hostname/system_name/")
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
        
        # Read the CSV files
        dvdl_csv_path = os.path.join(full_path, 'dvdl.csv')
        fl_csv_path = os.path.join(full_path, 'fl.csv')
        dvdl_rmsd_csv_path = os.path.join(full_path, 'dvdl_rmsd.csv')
        fl_rmsd_csv_path = os.path.join(full_path, 'fl_rmsd.csv')
        
        if not os.path.exists(dvdl_csv_path):
            print(f"error: data not found {dvdl_csv_path}")
            sys.exit(2)
            
        if not os.path.exists(fl_csv_path):
            print(f"error: data not found {fl_csv_path}")
            sys.exit(2)
        
        print("Calculating RMSD values...")
        # Read and process dvdl data
        dvdl = pd.read_csv(dvdl_csv_path)
        dvdl.columns = ['t_fs', 'lambda', 'phi_x', 'theta', 'v_theta', 'phi']
        dvdl = add_rmsd_values(dvdl, 'lambda', 'rmsd')
        
        # Read and process fl data
        fl = pd.read_csv(fl_csv_path)
        fl.columns = ["𝜆", "F(𝜆) fitted", "dF/d𝜆 fitted", "F(𝜆) raw", "<dU/d𝜆> raw"]
        fl = add_rmsd_values(fl, '𝜆', 'rmsd')
        
        # Save updated dataframes
        print(f"Saving RMSD data to {dvdl_rmsd_csv_path} and {fl_rmsd_csv_path}")
        dvdl.to_csv(dvdl_rmsd_csv_path, index=False)
        dvdl.to_parquet(f"{full_path}/dvdl_rmsd.parquet", index=False)
        fl.to_csv(fl_rmsd_csv_path, index=False)
        fl.to_parquet(f"{full_path}/fl_rmsd.parquet", index=False)
        
        # Create plots
        output_dir = "/scratch/users/lm18di/geometry/figures/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create interactive plot
        interactive_plot_path = os.path.join(output_dir, f"{system}_interactive_plot.html")
        create_interactive_plot(fl, interactive_plot_path)
        
        # Create static plots
        dvdl_pdf = os.path.join(output_dir, f"{system}_rmsd_time_plot.pdf")
        fl_pdf = os.path.join(output_dir, f"{system}_fl_rmsd_plot.pdf")
        
        rmsd_lvt_plot(dvdl, dvdl_pdf, system)
        rmsd_fl_plot(fl, fl_pdf, system) 