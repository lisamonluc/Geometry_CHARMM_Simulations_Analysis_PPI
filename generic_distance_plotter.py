#!/usr/bin/env python
"""
Generic distance data plotter.

Converts space-separated data files (frame distance) in a folder to CSV format
and generates a combined plot of distance vs. frame number for all files.

Usage example:
    python generic_distance_plotter.py /path/to/your/data_folder distance_plot.png

    This will process all compatible data files in the data_folder,
    convert them to CSV format (e.g., file.dat -> file.dat.csv),
    and create a combined plot saved as '/path/to/your/data_folder/distance_plot.png'.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

def convert_file_to_csv(input_file, output_file):
    """Convert a space-separated file to CSV format (frame,distance)"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("frame,distance\n") # Write header
            for line in f_in:
                # Split by whitespace and filter empty strings
                parts = [part for part in line.strip().split() if part]
                if len(parts) >= 2:
                    try:
                        # Validate that parts are numbers
                        frame_num = int(parts[0])
                        distance = float(parts[1])
                        f_out.write(f"{frame_num},{distance}\n")
                    except ValueError:
                        print(f"Warning: Skipping non-numeric line in {input_file}: {line.strip()}", file=sys.stderr)
                elif line.strip(): # Report lines that are not empty but don't have at least 2 parts
                     print(f"Warning: Skipping line with insufficient columns in {input_file}: {line.strip()}", file=sys.stderr)
        return True
    except UnicodeDecodeError:
        print(f"Warning: Skipping binary or non-UTF-8 file: {input_file}", file=sys.stderr)
        # Clean up potentially partially written csv file
        if os.path.exists(output_file):
            os.remove(output_file)
        return False
    except Exception as e:
        print(f"Error converting {input_file} to CSV: {e}", file=sys.stderr)
        # Clean up potentially partially written csv file
        if os.path.exists(output_file):
            os.remove(output_file)
        return False


def create_combined_plot(csv_files, output_plot_file):
    """Create a combined plot from multiple CSV files."""
    plt.figure(figsize=(12, 8))

    # Define a list of colorblind-friendly colors
    colorblind_friendly_colors = [
        '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
        '#56B4E9', '#F0E442', '#999999', '#000000'
    ]

    lines = []
    labels = []

    for i, csv_file in enumerate(sorted(csv_files)):
        base_name = os.path.basename(csv_file).replace('.csv', '')
        frame_nums = []
        distances = []

        try:
            with open(csv_file, 'r') as f:
                next(f) # Skip header row
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            frame_nums.append(int(parts[0]))
                            distances.append(float(parts[1]))
                        except ValueError:
                             print(f"Warning: Skipping non-numeric data row in {csv_file}: {line.strip()}", file=sys.stderr)

            if not frame_nums or not distances:
                print(f"Warning: No valid data found in {csv_file}. Skipping plot.", file=sys.stderr)
                continue

            # Plot the data with a unique color
            color_idx = i % len(colorblind_friendly_colors)
            line, = plt.plot(frame_nums, distances, color=colorblind_friendly_colors[color_idx], alpha=0.8)
            lines.append(line)
            labels.append(base_name)

        except Exception as e:
            print(f"Error reading or plotting {csv_file}: {e}", file=sys.stderr)

    if not lines:
        print("Error: No data could be plotted.", file=sys.stderr)
        plt.close() # Close the empty figure
        return

    # Set plot labels and title
    plt.xlabel("Frame Number")
    plt.ylabel("Distance (Å)") # Assuming Angstroms, adjust if needed
    plt.title("Distance vs. Frame Number")

    # Add legend to the right side of the plot
    plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust right margin for legend

    # Save the plot
    try:
        plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {output_plot_file}")
    except Exception as e:
        print(f"Error saving plot {output_plot_file}: {e}", file=sys.stderr)

    plt.close() # Close the plot figure


def process_folder(input_dir, output_plot_file):
    """Process files in the input directory, convert to CSV, and create a plot."""
    csv_files_to_plot = []
    processed_something = False

    print(f"Processing folder: {input_dir}")

    for item in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, item)

        # Skip directories and hidden files
        if os.path.isdir(input_file_path) or item.startswith('.'):
            continue

        # Avoid processing already generated csv or plot files
        if item.endswith(('.csv', '.png', '.pdf', '.jpg', '.jpeg')):
             print(f"Skipping potential output file: {item}")
             continue

        # Define CSV output path (place it alongside the original file)
        csv_output_path = input_file_path + '.csv'

        print(f"Attempting to convert: {item}")
        if convert_file_to_csv(input_file_path, csv_output_path):
            csv_files_to_plot.append(csv_output_path)
            processed_something = True
        else:
             # Conversion failed, warning already printed by convert function
             pass # Continue to next file

    # Create the combined plot if any files were successfully converted
    if csv_files_to_plot:
        print(f"Creating combined plot from {len(csv_files_to_plot)} CSV file(s)...")
        create_combined_plot(csv_files_to_plot, output_plot_file)
    elif processed_something:
         print("No valid data files were successfully converted to CSV for plotting.")
    else:
        print("No suitable files found to process in the specified directory.")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert space-separated data files to CSV and create a combined distance plot saved within the input folder.',
        formatter_class=argparse.RawDescriptionHelpFormatter # Preserve formatting in description
    )
    parser.add_argument('input_dir', help='Path to the folder containing data files.')
    parser.add_argument('output_plot_filename', help='Filename for the output plot (e.g., distance_plot.png), which will be saved inside the input directory.')

    args = parser.parse_args()

    # Check if the input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: The input directory '{args.input_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Construct the full output path inside the input directory
    # Use os.path.basename to ensure only the filename part is used from the argument
    output_filename = os.path.basename(args.output_plot_filename)
    full_output_plot_path = os.path.join(args.input_dir, output_filename)


    # Ensure the output plot filename has a reasonable extension
    if not any(output_filename.lower().endswith(ext) for ext in ['.png', '.pdf', '.jpg', '.jpeg', '.svg']):
         print(f"Warning: Output plot filename '{output_filename}' does not have a standard image extension (.png, .pdf, .jpg, .svg). Matplotlib might default to PNG.", file=sys.stderr)


    # Process the folder, passing the constructed full path for the plot
    process_folder(args.input_dir, full_output_plot_path)

if __name__ == "__main__":
    main() 