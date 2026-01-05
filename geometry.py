#!/usr/bin/python3

# usage: process_geo.py hosts/hostname/system_name/

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# dvdl.dat header line
# Step              Lambda                phiX               Theta              vTheta                 Phi  ...


def check_geo_dir(full_path):
    fp = Path(full_path)
    path_parts = fp.parts
    ok = 0

    if len(path_parts) < 3:
        ok = -1
        return (ok, "error: path too short")
    else:
        host = path_parts[1]
        system = path_parts[2]

        dvdl_path = os.path.join(full_path, "dvdl.dat")
        fl_path = os.path.join(full_path, "fl.dat")

        if not os.path.exists(dvdl_path):
            ok = -2
            return (ok, f"error: data not found {dvdl_path}")

        if not os.path.exists(fl_path):
            ok = -2
            err(err, f"error: data not found {fl_path}")

    ok = 1
    return (ok, "")


# takes a list of dataframes that may include restarts
# trims the end of a df that overlaps with its restart.
def trim_overlap_dvdl(df_list):

    dft = []
    for i in range(len(df_list) - 1):
        df_temp = df_list[i]
        df_temp = df_temp[df_temp["t_fs"] < df_list[i + 1]["t_fs"].iloc[0]]
        dft.append(df_temp)
    dft.append(df_list[-1])

    all_df = pd.concat(dft)

    return all_df

def trim_overlap_fl(df_list):
    dft = []
    for i in range(len(df_list) - 1):
        df_temp = df_list[i]
        # Keep data where lambda is greater than the next file's last lambda value
        # This preserves data from earlier files that extends beyond later files
        df_temp = df_temp[df_temp["𝜆"] > df_list[i + 1]["𝜆"].iloc[-1]]
        dft.append(df_temp)
    dft.append(df_list[-1])

    # Concatenate and sort by lambda value
    all_df = pd.concat(dft)
    all_df = all_df.sort_values("𝜆")
    
    # Remove any duplicate lambda values (keeping the last occurrence)
    all_df = all_df.drop_duplicates(subset=["𝜆"], keep='last')
    
    # Ensure we have continuous data by interpolating any gaps
    all_df = all_df.interpolate(method='linear', limit_direction='both')
    
    return all_df


# read the specified dvdl.dat
# numcols defaults to the 6 named columns
# if numcols is "all" all 36 columns will be used
def read_dvdl(dvdl_path, numcols=6):
    if numcols == "all":
        numcols = 36
    elif type(numcols) != int or numcols < 1 or numcols > 36:
        print("improper value for numcols")
        sys.exit(3)

    named_cols = [
        "t_fs",
        "lambda",
        "du/dl",
        "theta",
        "v_theta",
        "phi",
    ]

    # generate names for columns 7-36
    anon_cols = [f"col{n}" for n in range(7, 37)]
    all_cols = named_cols + anon_cols

    # get the lists of column numbers and names
    col_names = all_cols[:numcols]
    col_nums = list(range(numcols))

    dvdl = pd.read_csv(
        dvdl_path,
        sep=r"\s+",
        skiprows=1,
        usecols=col_nums,
    )
    print(dvdl.head())
    dvdl.columns = col_names
    # dvdl.columns = all_cols
    return dvdl


# returns the overlap trimmed concatenation of all
# of the dvdl.dat files, including restarts in the
# specified geometry directory
def read_dvdls(geo_dir, numcols=6):
    base_dir = Path(geo_dir)

    dvdls = []
    for dvdl_path in base_dir.glob("**/dvdl.dat"):
        dvdl_temp = read_dvdl(dvdl_path, numcols=numcols)
        dvdls.append(dvdl_temp)

    dvdl_df = trim_overlap_dvdl(dvdls)
    return dvdl_df


def read_fl(fl_file):
    # Skip both the #STEP line and the column names line
    fl = pd.read_csv(fl_file, sep=r"\s+", skiprows=2)
    # Assign our own column names
    fl.columns = ["𝜆", "F(𝜆) fitted", "dF/d𝜆 fitted", "F(𝜆) raw", "<dU/d𝜆> raw"]
    return fl


# returns the overlap trimmed concatenation of all
# of the fl.dat files, including restarts in the
# specified geometry directory
def read_fls(geo_dir):
    base_dir = Path(geo_dir)

    fls = []
    for fl_path in base_dir.glob("**/fl.dat"):
        fl_temp = read_fl(fl_path)
        fls.append(fl_temp)

    fl_df = trim_overlap_fl(fls)
    return fl_df


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
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], "b-", linewidth=1.5)
        # ax.plot(df['Time (fs)'], df['Lambda'], 'b-', linewidth=1.5)

        # Set labels and title with larger font size
        plt.xlabel("Time (fs)", fontsize=12, labelpad=10)
        plt.ylabel("𝜆", fontsize=12, labelpad=10)
        plt.title(f"𝜆 vs Time for {system}", fontsize=14)

        # Adjust layout to make room for labels
        plt.tight_layout(pad=3.0)

        # Save the figure to PDF with higher DPI for better quality
        pdf.savefig(bbox_inches="tight")

        # Close the figure to free memory
        plt.close()

    print(f"Plot saved to {plot_fname}")


def fl_plot(df, plot_fname, system):
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
        ax.plot(df.iloc[:, 0], df.iloc[:, 3], "r-", linewidth=1.5)
        # grabbing the first and fourth columns

        # Set labels and title with larger font size
        plt.xlabel("𝜆", fontsize=12, labelpad=10)
        plt.ylabel("∆G (Kcal/mol)", fontsize=12, labelpad=10)
        plt.title(f"FEL for {system}", fontsize=14)

        # Adjust layout to make room for labels
        plt.tight_layout(pad=3.0)

        # Save the figure to PDF with higher DPI for better quality
        pdf.savefig(bbox_inches="tight")

        # Close the figure to free memory
        plt.close()

    print(f"Plot saved to {plot_fname}")


if __name__ == "__main__":
    # This code will only run when geometry.py is executed directly
    # Not when it's imported as a module
    if len(sys.argv) != 2:
        print("Usage: geometry.py hosts/hostname/system_name/")
        sys.exit(1)
        
    full_path = sys.argv[1]
    ok, err_msg = check_geo_dir(full_path)
    if ok < 1:
        print(f"{full_path} is not a geometry directory")
        print(err_msg)
        sys.exit(1)
        
    # Example usage of the functions
    dvdl = read_dvdls(full_path + "/dvdl.dat")
    fl = read_fls(full_path + "/fl.dat")
