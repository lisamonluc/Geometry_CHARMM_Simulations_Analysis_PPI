#!/Users/lisamonluc/anaconda3/envs/mdtraj_env/bin/python

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.align import alignto
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from matplotlib.patches import Patch

# === USER SETUP ===
# Change paths if needed
root_path = '/Users/lisamonluc/Documents/yang_lab/4th_yr_figures/trajectories'

# BN_BS_BASIN_1
# CSV_FOLDER = os.path.join(root_path, "bn_bs_basin_1/bn_bs_basin_1_data/site_1")
# OUTPUT_FOLDER = os.path.join(root_path, "bn_bs_basin_1/geometry_analysis")
# DCD_PATH = os.path.join(root_path, "bn_bs_basin_1/dcd/minimum_1_frames.dcd")
# PDB_PATH = os.path.join(root_path, "bn_bs_pdb/eq.pdb")

# BN_BS_BASIN_2
CSV_FOLDER = os.path.join(root_path, "bn_bs_basin_2/data/site_1")
OUTPUT_FOLDER = os.path.join(root_path, "bn_bs_basin_2/geometry_analysis")
DCD_PATH = os.path.join(root_path, "bn_bs_basin_2/dcd/minimum_2_frames.dcd")
PDB_PATH = os.path.join(root_path, "bn_bs_pdb/eq.pdb")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def compute_2d_rmsd(dcd_path, pdb_path, atom_selection="name CA", force_recalculate=False):
    """
    Computes the 2D RMSD matrix between all frames in a DCD file using a reference PDB.
    Only uses selected atoms (default: alpha carbons).
    Each frame pair is aligned (centered and rotated) before RMSD calculation.
    """
    output_matrix = os.path.join(OUTPUT_FOLDER, "2d_rmsd_matrix.csv")
    output_df = os.path.join(OUTPUT_FOLDER, "2d_rmsd_matrix.pkl")
    progress_file = os.path.join(OUTPUT_FOLDER, "rmsd_progress.pkl")
    
    # Check if we need to recalculate
    if not force_recalculate and os.path.exists(output_df):
        print("Loading existing RMSD matrix...")
        rmsd_df = pd.read_pickle(output_df)
        print("RMSD matrix loaded successfully")
    else:
        print("Loading trajectory for RMSD calculation...")
        start_time = time.time()
        
        u = mda.Universe(pdb_path, dcd_path)
        sel = u.select_atoms(atom_selection)
        
        n_frames = len(u.trajectory)
        print(f"Total frames to process: {n_frames}")
        print(f"Estimated number of comparisons: {(n_frames * (n_frames + 1)) // 2:,} (upper triangle only)")
        
        # Initialize or load progress
        if os.path.exists(progress_file):
            print("Found previous progress file. Loading...")
            progress_data = pd.read_pickle(progress_file)
            rmsd_matrix = progress_data['matrix']
            last_completed_frame = progress_data['last_frame']
            print(f"Resuming from frame {last_completed_frame + 1}")
        else:
            rmsd_matrix = np.zeros((n_frames, n_frames))
            last_completed_frame = -1

        print("\nCalculating pairwise RMSDs...")
        # Calculate RMSDs using MDAnalysis's built-in RMSD calculation
        for i in tqdm(range(last_completed_frame + 1, n_frames), desc="Processing frames"):
            u.trajectory[i]
            ref_coords = sel.positions.copy()  # Reference coordinates for this frame
            
            for j in range(i, n_frames):  # Only calculate upper triangle
                u.trajectory[j]
                # Calculate RMSD with automatic centering and optimal rotation
                rmsd = mda.analysis.rms.rmsd(sel.positions,
                                           ref_coords,
                                           center=True,      # Center to remove translation
                                           superposition=True)  # Perform optimal rotation
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd  # Fill in symmetric part
            
            # Save progress every 50 frames
            if (i + 1) % 50 == 0 or i == n_frames - 1:
                print(f"\nSaving progress at frame {i + 1}...")
                progress_data = {
                    'matrix': rmsd_matrix,
                    'last_frame': i
                }
                pd.to_pickle(progress_data, progress_file)
                print("Progress saved.")

        # Calculate total time
        total_time = (time.time() - start_time) / 60  # in minutes
        print(f"\nRMSD calculation completed in {total_time:.1f} minutes")

        # Save final matrix as CSV
        np.savetxt(output_matrix, rmsd_matrix, delimiter=",")
        
        # Convert to DataFrame and save as pickle
        rmsd_df = pd.DataFrame(rmsd_matrix)
        rmsd_df.to_pickle(output_df)
        
        # Clean up progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        print(f"2D RMSD matrix saved to {output_matrix}")
        print(f"2D RMSD DataFrame saved to {output_df}")

    # Generate heatmap
    print("Generating heatmap...")
    plt.figure(figsize=(12, 10))
    
    # Set tick intervals to 50 frames
    n_frames = len(rmsd_df)
    tick_interval = 50
    tick_positions = np.arange(0, n_frames, tick_interval)
    
    # Create heatmap with custom ticks
    sns.heatmap(rmsd_df, cmap="viridis")
    plt.xticks(tick_positions, tick_positions, rotation=45, ha='right')
    plt.yticks(tick_positions, tick_positions)
    
    plt.title("2D RMSD Matrix (Å) - CA atoms only")
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")
    output_heatmap = os.path.join(OUTPUT_FOLDER, "2d_rmsd_heatmap.png")
    plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2D RMSD heatmap saved to {output_heatmap}")
    
    return rmsd_df

# === FUNCTION 1: Load all CSVs ===
def load_distance_data(csv_folder, n_frames):
    """
    Loads all CSV files in the given folder and combines them into one dataframe.
    Only processes files ending with .csv extension.
    Assumes format: frame, distance and filename = Arg59_X.csv → residue_pair = Arg59_X
    Note: First row in CSV is from PDB structure, so we expect n_frames + 1 total rows
    """
    all_data = []
    processed_files = 0
    skipped_files = 0
    expected_frames = n_frames + 1  # Add 1 to account for PDB structure frame

    print(f"\nProcessing files in {csv_folder}")
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            try:
                pair_name = file.replace(".csv", "")
                print(f"Processing {file}...")
                
                # Read CSV without headers and name columns explicitly
                df = pd.read_csv(os.path.join(csv_folder, file), 
                               header=None, 
                               names=['frame', 'distance'])
                
                # Validate data format
                if len(df.columns) != 2:
                    print(f"Warning: {file} has unexpected number of columns. Skipping.")
                    skipped_files += 1
                    continue
                
                # Check frame count
                frame_count = df['frame'].nunique()
                if frame_count != expected_frames:
                    print(f"Warning: {file} has {frame_count} frames, expected {expected_frames}")
                    print("Checking for missing frames...")
                    
                    # Find missing frames
                    all_frames = set(range(0, expected_frames))
                    present_frames = set(df['frame'].unique())
                    missing_frames = all_frames - present_frames
                    
                    if missing_frames:
                        print(f"Missing frames: {sorted(missing_frames)}")
                        print("Adding missing frames with NaN distances...")
                        
                        # Create DataFrame for missing frames
                        missing_df = pd.DataFrame({
                            'frame': list(missing_frames),
                            'distance': [np.nan] * len(missing_frames)
                        })
                        
                        # Combine with existing data
                        df = pd.concat([df, missing_df], ignore_index=True)
                        df = df.sort_values('frame')
                    
                    print(f"Total frames after processing: {df['frame'].nunique()}")
                
                df["residue_pair"] = pair_name
                all_data.append(df)
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                skipped_files += 1
                continue

    if not all_data:
        raise ValueError("No valid CSV files were processed. Check the input folder and file formats.")
    
    print(f"\nProcessing complete:")
    print(f"- Successfully processed: {processed_files} files")
    print(f"- Skipped: {skipped_files} files")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Final validation of frame count
    final_frame_count = combined_df['frame'].nunique()
    if final_frame_count != expected_frames:
        print(f"\nWarning: Final combined data has {final_frame_count} frames, expected {expected_frames}")
        print("This might affect the contact frequency calculations.")
    
    return combined_df

# === FUNCTION 2: AIM 1 – Contact Frequency ===
def calculate_contact_frequencies(df, threshold=4.0):
    """
    For each Arg59–X pair, calculate the % of frames with distance < threshold.
    Uses different thresholds for different interaction types:
    - Hydrogen bonds: ≤ 3.5Å
    - van der Waals: ≤ 3.8Å
    - Salt bridges: ≤ 4.0Å
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Extract interaction mode from residue_pair
    df['mode'] = df['residue_pair'].str.split('_').str[-1]
    
    # Apply different thresholds based on interaction type
    df['contact'] = np.where(
        df['mode'] == 'hbond',
        df['distance'] <= 3.5,  # 3.5Å for hydrogen bonds
        np.where(
            df['mode'] == 'vdw',
            df['distance'] <= 3.8,  # 3.8Å for van der Waals
            df['distance'] <= 4.0   # 4.0Å for salt bridges
        )
    )
    
    contact_counts = df[df['contact']].groupby('residue_pair').size()
    total_frames = df['frame'].nunique()
    freq_df = (contact_counts / total_frames).sort_values(ascending=False).reset_index()
    freq_df.columns = ['residue_pair', 'contact_frequency']
    return freq_df

# === FUNCTION 3: AIM 2 – Switching Events ===
def calculate_switching_events(df, threshold=4.0):
    """
    Counts how often Arg59 switches barstar partners between adjacent frames.
    Uses different thresholds for different interaction types:
    - Hydrogen bonds: ≤ 3.5Å
    - van der Waals: ≤ 3.8Å
    - Salt bridges: ≤ 4.0Å
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Extract interaction mode from residue_pair
    df['mode'] = df['residue_pair'].str.split('_').str[-1]
    
    # Apply different thresholds based on interaction type
    df['contact'] = np.where(
        df['mode'] == 'hbond',
        df['distance'] <= 3.5,  # 3.5Å for hydrogen bonds
        np.where(
            df['mode'] == 'vdw',
            df['distance'] <= 3.8,  # 3.8Å for van der Waals
            df['distance'] <= 4.0   # 4.0Å for salt bridges
        )
    )
    
    matrix = df.pivot(index='frame', columns='residue_pair', values='contact').fillna(False)
    
    # Debug: Print matrix information
    print(f"\nDebug - Contact matrix information:")
    print(f"Shape: {matrix.shape} (frames × residue pairs)")
    print(f"Residue pairs: {matrix.columns.tolist()}")
    print(f"Frame range: {matrix.index.min()} to {matrix.index.max()}")
    print(f"Number of contacts: {matrix.values.sum()}")
    
    frames = sorted(matrix.index)
    switches = 0

    for i in range(len(frames) - 1):
        contacts_now = set(matrix.columns[matrix.loc[frames[i]]])
        contacts_next = set(matrix.columns[matrix.loc[frames[i+1]]])
        if contacts_now != contacts_next:
            switches += 1

    return switches, matrix

# === FUNCTION 4: AIM 3 – Rebinding Events ===
def count_rebinding_events(df, threshold=4.0):
    """
    Counts how many times each Arg59–X interaction goes from 'off' to 'on' (rebinding).
    Uses different thresholds for different interaction types:
    - Hydrogen bonds: ≤ 3.5Å
    - van der Waals: ≤ 3.8Å
    - Salt bridges: ≤ 4.0Å
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Extract interaction mode from residue_pair
    df['mode'] = df['residue_pair'].str.split('_').str[-1]
    
    rebinding_counts = defaultdict(int)

    for pair, subdf in df.groupby('residue_pair'):
        subdf = subdf.sort_values('frame')
        # Get the appropriate threshold based on interaction type
        if 'hbond' in pair:
            pair_threshold = 3.5  # 3.5Å for hydrogen bonds
        elif 'vdw' in pair:
            pair_threshold = 3.8  # 3.8Å for van der Waals
        else:
            pair_threshold = 4.0  # 4.0Å for salt bridges
            
        prev = False
        for dist in subdf['distance']:
            curr = dist <= pair_threshold
            if not prev and curr:
                rebinding_counts[pair] += 1
            prev = curr

    rebinding_df = pd.DataFrame.from_dict(rebinding_counts, orient='index', columns=['rebinding_count'])
    rebinding_df = rebinding_df.sort_values('rebinding_count', ascending=False).reset_index()
    rebinding_df.columns = ['residue_pair', 'rebinding_count']
    return rebinding_df

def plot_contact_frequencies(freq_df):
    """Create a bar plot of contact frequencies"""
    plt.figure(figsize=(12, 6))
    
    # Extract interaction mode from residue_pair
    freq_df['mode'] = freq_df['residue_pair'].str.split('_').str[-1]
    freq_df['residue2'] = freq_df['residue_pair'].str.split('_').str[1]
    
    # Check for entries without proper mode suffix
    valid_modes = {'salt', 'hbond', 'vdw'}
    invalid_entries = freq_df[~freq_df['mode'].isin(valid_modes)]
    if not invalid_entries.empty:
        print("\nWarning: Found entries without proper interaction mode suffix (_salt, _hbond, or _vdw):")
        for _, row in invalid_entries.iterrows():
            print(f"  - {row['residue_pair']}")
        print("Fix CSV file naming to include mode of interaction.")
        raise ValueError("Invalid CSV file names detected. Please fix naming convention.")
    
    # Calculate total contact frequency and no-contact frequency
    total_contact_freq = freq_df['contact_frequency'].sum()
    no_contact_freq = max(0, 1.0 - total_contact_freq)  # Ensure non-negative
    
    # Create a DataFrame row for no contacts
    no_contact_row = pd.DataFrame({
        'residue_pair': ['No_Contacts'],
        'contact_frequency': [no_contact_freq],
        'mode': ['none'],
        'residue2': ['No Contacts']
    })
    
    # Combine with original data
    plot_df = pd.concat([freq_df, no_contact_row], ignore_index=True)
    
    # Create color mapping for different modes
    mode_colors = {
        'salt': '#D55E00',  # Orange-red
        'hbond': '#56B4E9', # Light blue
        'vdw': '#006837',   # Dark green
        'none': '#999999'   # Gray for no contacts
    }
    
    # Plot bars with different colors for each mode
    bars = []
    labels = []
    for mode, color in mode_colors.items():
        mode_data = plot_df[plot_df['mode'] == mode]
        if not mode_data.empty:
            bar = plt.bar(mode_data['residue2'], mode_data['contact_frequency'], 
                         color=color, alpha=0.7)
            bars.extend(bar)
            labels.extend([mode] * len(bar))
        else:
            # Add an invisible bar to ensure the legend shows all modes
            bar = plt.bar([0], [0], color=color, alpha=0.7, visible=False)
            bars.append(bar)
            labels.append(mode)
    
    # Add horizontal line at 1.0
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.text(0, 1.02, 'Total possible contacts = 1.0', color='gray', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Contact Frequency')
    plt.title('Contact Frequencies of Arg59 with Different Residues')
    
    # Update legend labels
    legend_labels = {
        'salt': 'Salt Bridge',
        'hbond': 'Hydrogen Bond',
        'vdw': 'van der Waals',
        'none': 'No Contacts'
    }
    
    # Create legend with all entries, even if some are not in the data
    legend_handles = [plt.Rectangle((0,0),1,1, color=mode_colors[mode], alpha=0.7) 
                     for mode in mode_colors.keys()]
    legend_texts = [legend_labels[mode] for mode in mode_colors.keys()]
    
    plt.legend(legend_handles, legend_texts,
              title='Interaction Mode',
              bbox_to_anchor=(1.15, 1))
    
    # Set y-axis limits to ensure all bars are visible
    plt.ylim(0, 1.1)  # From 0 to 1.1 to show the full bars and the line at 1.0
    
    plt.tight_layout()
    
    output_plot = os.path.join(OUTPUT_FOLDER, "contact_frequencies_plot.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Contact frequencies plot saved to {output_plot}")
    
    # Print the frequencies including no contacts
    print("\nContact Frequencies Summary:")
    for _, row in plot_df.iterrows():
        print(f"{row['residue2']}: {row['contact_frequency']:.3f}")
    print(f"Total (should be 1.0): {plot_df['contact_frequency'].sum():.3f}")

def plot_rebinding_counts(rebinding_df):
    """Create two versions of the rebinding counts plot - one with grid lines and one without"""
    # Version 1: With grid lines and count numbers
    plt.figure(figsize=(12, 6))
    
    # Extract interaction mode from residue_pair
    rebinding_df['mode'] = rebinding_df['residue_pair'].str.split('_').str[-1]
    rebinding_df['residue2'] = rebinding_df['residue_pair'].str.split('_').str[1]
    
    # Create color mapping for different modes
    mode_colors = {
        'salt': '#D55E00',  # Orange-red
        'hbond': '#56B4E9', # Light blue
        'vdw': '#006837'    # Dark green
    }
    
    # Plot bars with different colors for each mode
    for mode, color in mode_colors.items():
        mode_data = rebinding_df[rebinding_df['mode'] == mode]
        if not mode_data.empty:
            bars = plt.bar(mode_data['residue2'], mode_data['rebinding_count'], 
                    color=color, label=mode, alpha=0.7)
            # Add count numbers on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
        else:
            # Add an invisible bar to ensure the legend shows all modes
            plt.bar([0], [0], color=color, label=mode, alpha=0.7, visible=False)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Rebinding Events')
    plt.title('Rebinding Events of Arg59 with Different Residues (with grid lines)')
    
    # Update legend labels
    legend_labels = {
        'salt': 'Salt Bridge',
        'hbond': 'Hydrogen Bond',
        'vdw': 'van der Waals'
    }
    
    # Create legend with all entries, even if some are not in the data
    legend_handles = [plt.Rectangle((0,0),1,1, color=mode_colors[mode], alpha=0.7) 
                     for mode in mode_colors.keys()]
    legend_texts = [legend_labels[mode] for mode in mode_colors.keys()]
    
    # Place legend in the upper right corner of the plot
    plt.legend(legend_handles, legend_texts,
              title='Interaction Mode',
              loc='upper right',
              bbox_to_anchor=(0.98, 0.98),
              frameon=True,
              framealpha=0.9)
    
    # Add grid lines
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_plot = os.path.join(OUTPUT_FOLDER, "rebinding_counts_plot_with_grid.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rebinding counts plot with grid lines saved to {output_plot}")
    
    # Version 2: Without grid lines
    plt.figure(figsize=(12, 6))
    
    # Plot bars with different colors for each mode
    for mode, color in mode_colors.items():
        mode_data = rebinding_df[rebinding_df['mode'] == mode]
        if not mode_data.empty:
            plt.bar(mode_data['residue2'], mode_data['rebinding_count'], 
                    color=color, label=mode, alpha=0.7)
        else:
            # Add an invisible bar to ensure the legend shows all modes
            plt.bar([0], [0], color=color, label=mode, alpha=0.7, visible=False)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Rebinding Events')
    plt.title('Rebinding Events of Arg59 with Different Residues (without grid lines)')
    
    # Create legend with all entries, even if some are not in the data
    plt.legend(legend_handles, legend_texts,
              title='Interaction Mode',
              loc='upper right',
              bbox_to_anchor=(0.98, 0.98),
              frameon=True,
              framealpha=0.9)
    
    # No grid lines in this version
    plt.grid(False)
    
    plt.tight_layout()
    
    output_plot = os.path.join(OUTPUT_FOLDER, "rebinding_counts_plot_without_grid.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rebinding counts plot without grid lines saved to {output_plot}")

def plot_contact_matrix(contact_matrix):
    """Create a heatmap of the contact matrix"""
    plt.figure(figsize=(10, 14))  # Make figure taller to accommodate legend below
    
    # Create custom legend patches with larger size
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Contact (distance < 4Å)'),
        Patch(facecolor='white', edgecolor='black', label='No Contact (distance ≥ 4Å)')
    ]
    
    # Create the heatmap with binary colors
    ax = plt.gca()
    sns.heatmap(contact_matrix, 
                cmap=['white', 'black'],  # White for 0, Black for 1
                cbar=False,  # Remove the colorbar since we're using a custom legend
                xticklabels=contact_matrix.columns,  # Show residue pair names
                yticklabels=True)  # Show frame numbers
    
    # Set y-axis (frame) tick intervals to 50 frames
    n_frames = len(contact_matrix)
    tick_interval = 50
    tick_positions = np.arange(0, n_frames, tick_interval)
    plt.yticks(tick_positions, tick_positions)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.xlabel('Residue Pairs')
    plt.ylabel('Frame Number')
    plt.title('Contact Matrix for Arg59 Interactions')
    
    # Add custom legend below the plot with larger font size
    ax.legend(handles=legend_elements, 
              bbox_to_anchor=(0.5, -0.2),  # Center below plot
              loc='upper center',
              frameon=True,
              edgecolor='black',
              fontsize=12,  # Larger font
              ncol=2)  # Two columns side by side
    
    # Adjust layout to prevent label cutoff and leave space for legend
    plt.tight_layout()
    
    output_plot = os.path.join(OUTPUT_FOLDER, "contact_matrix_heatmap.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight', pad_inches=0.5)  # Add padding for legend
    plt.close()
    print(f"Contact matrix heatmap saved to {output_plot}")

def find_no_contact_frames(contact_matrix):
    """
    Find frames where there are no contacts (all distances > 4Å) and analyze gaps.
    
    Parameters:
    -----------
    contact_matrix : pandas DataFrame or numpy array
        The binary contact matrix where 1 = contact, 0 = no contact
    
    Returns:
    --------
    tuple
        - Frame numbers where no contacts were observed
        - List of (start_frame, end_frame, gap_size) for largest gaps
    """
    # Convert all rows to sum of contacts
    contact_sums = contact_matrix.sum(axis=1)
    # Find indices where sum is 0 (no contacts)
    no_contact_frames = np.where(contact_sums == 0)[0]
    
    # Analyze gaps between no-contact frames
    gaps = []
    if len(no_contact_frames) > 1:
        gap_sizes = np.diff(no_contact_frames)
        for i in range(len(gap_sizes)):
            gap_start = no_contact_frames[i]
            gap_end = no_contact_frames[i + 1]
            gap_size = gap_sizes[i]
            gaps.append((gap_start, gap_end, gap_size))
        
        # Sort gaps by size in descending order
        gaps.sort(key=lambda x: x[2], reverse=True)
    
    return no_contact_frames, gaps

# === MAIN FUNCTION: Run all three aims ===
def main():
    print("Delete 2d rmsd matrix dataframe if you want to recompute it.")
    
    # Get number of frames from DCD file once
    print("\nLoading trajectory information...")
    u = mda.Universe(PDB_PATH, DCD_PATH)
    n_frames = len(u.trajectory)
    print(f"Detected {n_frames} frames in DCD file")
    
    print("\nLoading distance data...")
    df = load_distance_data(CSV_FOLDER, n_frames)

    print("\n[AIM 1] Calculating contact frequencies...")
    freq_df = calculate_contact_frequencies(df)
    print(freq_df)
    output_freq = os.path.join(OUTPUT_FOLDER, "arg59_contact_frequencies.csv")
    freq_df.to_csv(output_freq, index=False)
    print(f"Contact frequencies saved to {output_freq}")
    plot_contact_frequencies(freq_df)

    print("\n[AIM 2] Calculating switching events...")
    switch_count, contact_matrix = calculate_switching_events(df)
    print(f"Total switching events: {switch_count}")
    output_switch = os.path.join(OUTPUT_FOLDER, "arg59_contact_matrix_binary.csv")
    contact_matrix.to_csv(output_switch)
    print(f"Contact matrix saved to {output_switch}")
    plot_contact_matrix(contact_matrix)
    
    # Find frames with no contacts
    no_contact_frames, gaps = find_no_contact_frames(contact_matrix)
    print(f"\nFound {len(no_contact_frames)} frames with no contacts:")
    if len(no_contact_frames) > 0:
        print(f"Frame numbers: {no_contact_frames.tolist()}")
        # Save the frame numbers to a file
        output_frames = os.path.join(OUTPUT_FOLDER, "no_contact_frames.txt")
        np.savetxt(output_frames, no_contact_frames, fmt='%d')
        print(f"No-contact frame numbers saved to {output_frames}")
        
        # Calculate percentage of frames with no contacts
        percent_no_contacts = (len(no_contact_frames) / len(contact_matrix)) * 100
        print(f"Percentage of frames with no contacts: {percent_no_contacts:.2f}%")
        
        # Analyze gaps between no-contact frames
        if gaps:
            print("\nGaps between no-contact frames:")
            print(f"Total number of gaps: {len(gaps)}")
            
            # Basic gap statistics
            gap_sizes = [g[2] for g in gaps]
            print(f"Minimum gap: {min(gap_sizes)} frames")
            print(f"Maximum gap: {max(gap_sizes)} frames")
            print(f"Average gap: {np.mean(gap_sizes):.2f} frames")
            print(f"Median gap: {np.median(gap_sizes):.0f} frames")
            
            # Report the top 5 largest gaps
            print("\nTop 5 largest gaps between no-contact frames:")
            for i, (start, end, size) in enumerate(gaps[:5], 1):
                print(f"{i}. Frames {start} → {end} (gap size: {size} frames)")
            
            # Save gap analysis to file
            output_gaps = os.path.join(OUTPUT_FOLDER, "no_contact_gaps.txt")
            with open(output_gaps, 'w') as f:
                f.write("start_frame,end_frame,gap_size\n")
                for start, end, size in gaps:
                    f.write(f"{start},{end},{size}\n")
            print(f"\nDetailed gap analysis saved to {output_gaps}")

    print("\n[AIM 3] Counting rebinding events...")
    rebinding_df = count_rebinding_events(df)
    print(rebinding_df)
    output_rebind = os.path.join(OUTPUT_FOLDER, "arg59_rebinding_counts.csv")
    rebinding_df.to_csv(output_rebind, index=False)
    print(f"Rebinding counts saved to {output_rebind}")
    plot_rebinding_counts(rebinding_df)

    print("\n[OPTIONAL] Computing 2D RMSD matrix across frames...")
    # Load or compute RMSD matrix
    rmsd_df = compute_2d_rmsd(DCD_PATH, PDB_PATH, force_recalculate=False)

    print("✅ Analysis complete.")

# Run the script
if __name__ == "__main__":
    main()
