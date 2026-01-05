#!/usr/bin/env python
"""
Distance data processor for site folders.

Usage example:
    python process_distance_data.py /path/to/your/data_folder 1 # Change minima number if needed, here it's 1

    This will process all data files in site_1, site_2, etc. folders within the data_folder,
    convert them to CSV format, and create plots with data grouped by site and interaction mode.
    The plots will be saved as 'distance_plots_minima_1.png' and 'distance_plots_by_mode_minima_1.png' in the data_folder.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

def convert_file_to_csv(input_file, output_file):
    """Convert a space-separated file to CSV format"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w') as f:
        for line in lines:
            # Split by whitespace and join with comma
            parts = line.strip().split()
            if len(parts) >= 2:
                frame_num = parts[0]
                distance = parts[1]
                f.write(f"{frame_num},{distance}\n")

def create_plot(site_csv_files, minima_number, output_file):
    """Create a plot from the CSV files with colors grouped by site"""
    plt.figure(figsize=(12, 8))
    
    # Define a list of colorblind-friendly colors
    # Based on colorblind-friendly palette: blue, red, green, purple, orange, yellow
    colorblind_friendly_colors = [
        '#0072B2',  # blue
        '#D55E00',  # red/orange
        '#009E73',  # green
        '#CC79A7',  # pink/purple
        '#E69F00',  # orange/yellow
        '#56B4E9',  # light blue
        '#F0E442',  # yellow
        '#999999',  # gray
    ]
    
    # Assign colors to sites
    site_colors = {}
    for i, site_name in enumerate(sorted(site_csv_files.keys())):
        color_idx = i % len(colorblind_friendly_colors)  # Cycle through colors if more sites than colors
        site_colors[site_name] = colorblind_friendly_colors[color_idx]
    
    # Store plot lines and labels for legend sorting
    lines = []
    labels = []
    
    # Plot all files, coloring by site
    for site_name in sorted(site_csv_files.keys()):
        site_color = site_colors[site_name]
        csv_files = site_csv_files[site_name]
        
        for csv_file in sorted(csv_files):
            # Extract the filename without extension for the legend
            base_name = os.path.basename(csv_file).replace('.csv', '')
            
            # Read the CSV file
            frame_nums = []
            distances = []
            with open(csv_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        frame_nums.append(int(parts[0]))
                        distances.append(float(parts[1]))
            
            # Plot the data with the site's color
            line, = plt.plot(frame_nums, distances, color=site_color)
            lines.append(line)
            labels.append(f"{site_name}: {base_name}")
    
    # Set plot labels and title
    plt.xlabel(f"Frame Number within Minima {minima_number}")
    plt.ylabel("Distance (Å)")
    plt.title(f"Distance Plot for Minima {minima_number} (Grouped by Site)")
    
    # Add legend to the right side of the plot
    plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Save the plot
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

def create_imf_plots(site_csv_files, minima_number, output_dir):
    """Create individual plots for each data file and one combined plot, with colors based on filenames"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define a list of colorblind-friendly colors
    colorblind_friendly_colors = [
        '#0072B2',  # blue
        '#D55E00',  # red/orange
        '#009E73',  # green
        '#CC79A7',  # pink/purple
        '#E69F00',  # orange/yellow
        '#56B4E9',  # light blue
        '#F0E442',  # yellow
        '#999999',  # gray
    ]
    
    # Collect all CSV files into a sorted list
    all_csv_files = []
    for site_name in sorted(site_csv_files.keys()):
        for csv_file in sorted(site_csv_files[site_name]):
            all_csv_files.append((site_name, csv_file))
    
    # Create individual plots
    for i, (site_name, csv_file) in enumerate(all_csv_files):
        plt.figure(figsize=(10, 6))
        
        # Extract the filename without extension for the title
        base_name = os.path.basename(csv_file).replace('.csv', '')
        
        # Read the CSV file
        frame_nums = []
        distances = []
        with open(csv_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    frame_nums.append(int(parts[0]))
                    distances.append(float(parts[1]))
        
        # Plot the data
        plt.plot(frame_nums, distances, color=colorblind_friendly_colors[i % len(colorblind_friendly_colors)])
        
        # Set plot labels and title
        plt.xlabel(f"Frame Number within Minima {minima_number}")
        plt.ylabel("Distance (Å)")
        plt.title(f"{base_name} - Minima {minima_number}")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the individual plot
        individual_plot_path = os.path.join(output_dir, f"{base_name}_minima_{minima_number}.png")
        plt.savefig(individual_plot_path, dpi=300)
        plt.close()
        print(f"Individual plot saved to {individual_plot_path}")
    
    # Create combined plot
    plt.figure(figsize=(12, 8))
    lines = []
    labels = []
    
    for i, (site_name, csv_file) in enumerate(all_csv_files):
        # Extract the filename without extension for the legend
        base_name = os.path.basename(csv_file).replace('.csv', '')
        
        # Read the CSV file
        frame_nums = []
        distances = []
        with open(csv_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    frame_nums.append(int(parts[0]))
                    distances.append(float(parts[1]))
        
        # Plot the data with a unique color
        line, = plt.plot(frame_nums, distances, color=colorblind_friendly_colors[i % len(colorblind_friendly_colors)])
        lines.append(line)
        labels.append(f"{site_name}: {base_name}")
    
    # Set plot labels and title
    plt.xlabel(f"Frame Number within Minima {minima_number}")
    plt.ylabel("Distance (Å)")
    plt.title(f"Combined Distance Plot for Minima {minima_number}")
    
    # Add legend to the right side of the plot
    plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Save the combined plot
    combined_plot_path = os.path.join(output_dir, f"combined_distance_plots_minima_{minima_number}.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.close()
    print(f"Combined plot saved to {combined_plot_path}")

def create_plot_by_mode(site_csv_files, minima_number, output_file):
    """Create a plot where data is grouped by interaction mode (salt, hbond, vdw)"""
    plt.figure(figsize=(12, 8))
    
    # Define colors for each interaction mode
    mode_colors = {
        'salt': '#D55E00',  # red/orange
        'hbond': '#0072B2',  # blue
        'vdw': '#009E73'    # green
    }
    
    # Store plot lines and labels for legend
    lines = []
    labels = []
    
    # Group files by interaction mode
    mode_files = {'salt': [], 'hbond': [], 'vdw': []}
    for site_name in site_csv_files:
        for csv_file in site_csv_files[site_name]:
            # Extract interaction mode and residue names from filename
            base_name = os.path.basename(csv_file).replace('.csv', '')
            parts = base_name.split('_')
            mode = parts[-1]  # Last part is the mode (salt, hbond, vdw)
            if len(parts) >= 3:  # Make sure we have enough parts to extract residues
                residue1 = parts[0]  # First residue
                residue2 = parts[1]  # Second residue
                if mode in mode_files:
                    mode_files[mode].append((site_name, csv_file, f"{residue1} - {residue2}"))
    
    # Plot each mode's data
    for mode, files in mode_files.items():
        for site_name, csv_file, residue_pair in files:
            # Read the CSV file
            frame_nums = []
            distances = []
            with open(csv_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        frame_nums.append(int(parts[0]))
                        distances.append(float(parts[1]))
            
            # Plot the data with the mode's color
            line, = plt.plot(frame_nums, distances, color=mode_colors[mode], alpha=0.7)
            lines.append(line)
            # Format the label as "mode: residue1 - residue2 (site X)"
            site_num = site_name.split('_')[-1]  # Extract site number
            labels.append(f"{mode}: {residue_pair} (site {site_num})")
    
    # Set plot labels and title
    plt.xlabel(f"Frame Number within Minima {minima_number}")
    plt.ylabel("Distance (Å)")
    plt.title(f"Distance Plot for Minima {minima_number} (Grouped by Interaction Mode)")
    
    # Add legend to the right side of the plot
    plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Save the plot
    plt.savefig(output_file, dpi=300)
    print(f"Mode-based plot saved to {output_file}")

def create_site_plot(site_name, csv_files, minima_number, output_file):
    """Create a plot for a single site with all its distance data"""
    plt.figure(figsize=(12, 8))
    
    # Define a list of colorblind-friendly colors
    colorblind_friendly_colors = [
        '#0072B2',  # blue
        '#D55E00',  # red/orange
        '#009E73',  # green
        '#CC79A7',  # pink/purple
        '#E69F00',  # orange/yellow
        '#56B4E9',  # light blue
        '#F0E442',  # yellow
        '#999999',  # gray
    ]
    
    # Store plot lines and labels for legend
    lines = []
    labels = []
    
    # Plot all files for this site
    for i, csv_file in enumerate(sorted(csv_files)):
        # Extract the filename without extension for the legend
        base_name = os.path.basename(csv_file).replace('.csv', '')
        
        # Read the CSV file
        frame_nums = []
        distances = []
        with open(csv_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    frame_nums.append(int(parts[0]))
                    distances.append(float(parts[1]))
        
        # Plot the data with a unique color
        color_idx = i % len(colorblind_friendly_colors)
        line, = plt.plot(frame_nums, distances, color=colorblind_friendly_colors[color_idx])
        lines.append(line)
        labels.append(base_name)
    
    # Set plot labels and title
    plt.xlabel(f"Frame Number within Minima {minima_number}")
    plt.ylabel("Distance (Å)")
    plt.title(f"Distance Plot for {site_name} - Minima {minima_number}")
    
    # Add legend to the right side of the plot
    plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Save the plot
    plt.savefig(output_file, dpi=300)
    print(f"Site plot saved to {output_file}")
    plt.close()

def process_directory(root_path, minima_number):
    """Process site folders in the root path"""
    site_csv_files = {}
    
    # Look for site_X folders in the root path
    for item in os.listdir(root_path):
        site_path = os.path.join(root_path, item)
        if os.path.isdir(site_path) and item.startswith("site_"):
            print(f"Processing {item} folder")
            site_name = item
            
            # Get all data files in this site folder
            data_files = []
            csv_files = []
            
            for file in os.listdir(site_path):
                full_path = os.path.join(site_path, file)
                # Skip directories, system files, and files that already have .csv or .png extension
                if (os.path.isfile(full_path) and 
                    not file.startswith('.') and  # Skip hidden files like .DS_Store
                    not file.endswith('.csv') and 
                    not file.endswith('.png')):
                    data_files.append(full_path)
            
            print(f"Found {len(data_files)} data files in {site_name}")
            
            # Convert each file to CSV
            for data_file in data_files:
                csv_file = data_file + '.csv'
                print(f"Converting {os.path.basename(data_file)} to CSV format")
                try:
                    convert_file_to_csv(data_file, csv_file)
                    csv_files.append(csv_file)
                except UnicodeDecodeError:
                    print(f"Warning: Skipping {data_file} as it appears to be a binary file")
                except Exception as e:
                    print(f"Warning: Error processing {data_file}: {str(e)}")
            
            if csv_files:
                site_csv_files[site_name] = csv_files
                # Create individual site plot
                site_plot_path = os.path.join(root_path, f"{site_name}_distance_plot_minima_{minima_number}.png")
                create_site_plot(site_name, csv_files, minima_number, site_plot_path)
    
    # Create the combined plots
    if site_csv_files:
        # Create plot with colors grouped by site
        output_plot = os.path.join(root_path, f"distance_plots_by_site_minima_{minima_number}.png")
        create_plot(site_csv_files, minima_number, output_plot)
        
        # Create plot grouped by interaction mode
        output_mode_plot = os.path.join(root_path, f"distance_plots_by_mode_minima_{minima_number}.png")
        create_plot_by_mode(site_csv_files, minima_number, output_mode_plot)
        
        # Create individual and combined IMF plots
        imf_plots_dir = os.path.join(root_path, f"imf_plots_minima_{minima_number}")
        create_imf_plots(site_csv_files, minima_number, imf_plots_dir)
    else:
        print("No files were processed. Check that the input path contains site_X folders with data files.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process distance data files from site folders and create plots')
    parser.add_argument('input_path', help='Path containing site_X folders with data files')
    parser.add_argument('minima_number', type=int, help='Minima number for plot labeling')
    
    args = parser.parse_args()
    
    # Check if the input path exists
    if not os.path.isdir(args.input_path):
        print(f"Error: The path {args.input_path} does not exist")
        sys.exit(1)
    
    # Process the directory structure
    process_directory(args.input_path, args.minima_number)

if __name__ == "__main__":
    main()
