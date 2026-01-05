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
CSV_FOLDER_1 = os.path.join(root_path, "bn_bs_basin_2/data/site_1")
CSV_FOLDER_2 = os.path.join(root_path, "bn_bs_basin_2/data/site_2")
CSV_FOLDER_3 = os.path.join(root_path, "bn_bs_basin_2/data/site_3")
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
def load_distance_data(csv_folders, n_frames):
    """
    Loads all CSV files from a list of folders and combines them into one dataframe.
    Assigns a site number based on the folder index.
    Only processes files ending with .csv extension.
    Assumes format: frame, distance and filename = Arg59_X.csv → residue_pair = Arg59_X_siteY
    Note: First row in CSV is from PDB structure, so we expect n_frames + 1 total rows
    """
    all_data = []
    processed_files_total = 0
    skipped_files_total = 0
    expected_frames = n_frames + 1  # Add 1 to account for PDB structure frame

    print(f"\nLoading data from {len(csv_folders)} site folders...")

    for site_index, csv_folder in enumerate(csv_folders):
        site_number = site_index + 1  # Site numbers start from 1
        print(f"\nProcessing files in {csv_folder} (Site {site_number})")
        processed_files_site = 0
        skipped_files_site = 0

        if not os.path.isdir(csv_folder):
            print(f"Warning: Folder not found - {csv_folder}. Skipping.")
            continue

        for file in os.listdir(csv_folder):
            if file.endswith(".csv"):
                try:
                    base_pair_name = file.replace(".csv", "")
                    # Append site number to the pair name
                    pair_name = f"{base_pair_name}_site{site_number}"
                    print(f"Processing {file} as {pair_name}...")
                    
                    # Read CSV without headers and name columns explicitly
                    df = pd.read_csv(os.path.join(csv_folder, file), 
                                   header=None, 
                                   names=['frame', 'distance'])
                    
                    # Validate data format
                    if len(df.columns) != 2:
                        print(f"Warning: {file} has unexpected number of columns. Skipping.")
                        skipped_files_site += 1
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
                    df["site"] = site_number  # Add site number column
                    all_data.append(df)
                    processed_files_site += 1
                    
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    skipped_files_site += 1
                    continue

        print(f"\nSite {site_number} processing complete:")
        print(f"- Successfully processed: {processed_files_site} files")
        print(f"- Skipped: {skipped_files_site} files")
        processed_files_total += processed_files_site
        skipped_files_total += skipped_files_site

    if not all_data:
        raise ValueError("No valid CSV files were processed. Check the input folders and file formats.")
    
    print(f"\nOverall processing complete:")
    print(f"- Successfully processed: {processed_files_total} files from {len(csv_folders)} sites")
    print(f"- Skipped: {skipped_files_total} files")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Final validation of frame count per site
    for site_num in combined_df['site'].unique():
        site_df = combined_df[combined_df['site'] == site_num]
        final_frame_count = site_df['frame'].nunique()
        if final_frame_count != expected_frames:
            print(f"\nWarning: Final combined data for Site {site_num} has {final_frame_count} frames, expected {expected_frames}")
            print("This might affect calculations.")
            
    # Extract mode and residue2 for easier access later
    combined_df['mode'] = combined_df['residue_pair'].str.split('_').str[-2] # Mode is now second to last
    combined_df['residue2'] = combined_df['residue_pair'].str.split('_').str[1] # Residue name is still second part
    # Base name without site - using apply for robustness
    combined_df['residue_pair_base'] = combined_df['residue_pair'].apply(lambda x: x.rsplit('_', 1)[0]) 

    return combined_df

# === FUNCTION 2: AIM 1 – Contact Frequency ===
def calculate_contact_frequencies(df):
    """
    For each Arg59–X pair, calculate the % of frames with distance < threshold.
    Uses different thresholds for different interaction types:
    - Hydrogen bonds: ≤ 3.5Å
    - van der Waals: ≤ 3.8Å
    - Salt bridges: ≤ 4.0Å
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Mode is already extracted in load_distance_data
    # df['mode'] = df['residue_pair'].str.split('_').str[-2] # Mode is second to last element
    
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
    # Calculate total frames per site, then average? Or just use overall total frames?
    # Let's stick to total unique frames across all sites for now.
    total_frames = df['frame'].nunique() 
    freq_df = (contact_counts / total_frames).sort_values(ascending=False).reset_index()
    freq_df.columns = ['residue_pair', 'contact_frequency']
    
    # Add back site, mode, residue2 for plotting convenience
    freq_df['site'] = freq_df['residue_pair'].str.split('_').str[-1].str.replace('site', '').astype(int)
    freq_df['mode'] = freq_df['residue_pair'].str.split('_').str[-2]
    freq_df['residue2'] = freq_df['residue_pair'].str.split('_').str[1]
    
    return freq_df

# === FUNCTION 3: AIM 2 – Switching Events ===
def calculate_switching_events(df):
    """
    Counts how often Arg59 switches barstar partners between adjacent frames.
    Creates a site-encoded matrix (min site number) for plotting.
    Creates individual binary contact matrices for each site for co-occurrence analysis.
    Uses different thresholds for different interaction types:
    - Hydrogen bonds: ≤ 3.5Å
    - van der Waals: ≤ 3.8Å
    - Salt bridges: ≤ 4.0Å
    
    Returns:
        switches (int): Total number of switching events.
        site_matrix (pd.DataFrame): Matrix for plotting (0=none, 1=site1, 2=site2, ...).
        site_binary_matrices (dict): Dict of {site_num: binary_matrix} for each site.
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Mode is already extracted
    # df['mode'] = df['residue_pair'].str.split('_').str[-2]
    
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
    
    # --- Calculate Switch Count (using binary contact matrix per unique residue_pair_site) ---
    # Pivot to get binary contact matrix (True/False) for each residue_pair (including site)
    binary_matrix_full = df.pivot(index='frame', columns='residue_pair', values='contact').fillna(False)
    
    frames = sorted(binary_matrix_full.index)
    switches = 0
    for i in range(len(frames) - 1):
        # Contacts active in the current frame (across all sites)
        contacts_now = set(binary_matrix_full.columns[binary_matrix_full.loc[frames[i]]])
        # Contacts active in the next frame (across all sites)
        contacts_next = set(binary_matrix_full.columns[binary_matrix_full.loc[frames[i+1]]])
        # A switch occurs if the set of active contacts (including site) changes
        if contacts_now != contacts_next:
            switches += 1

    # --- Create Site-Encoded Contact Matrix for Plotting (min site number) ---
    # --- AND Create Site-Specific Binary Matrices ---
    # Pivot to get distances first
    distance_matrix = df.pivot(index='frame', columns='residue_pair', values='distance')
    
    # Create the site-encoded matrix: index=frame, columns=residue_pair_base (Arg59_RESIDUE_MODE)
    base_pairs = sorted(df['residue_pair_base'].unique()) # Sort for consistent column order
    site_matrix = pd.DataFrame(0, index=frames, columns=base_pairs) # Initialize with 0 (no contact)
    
    # Initialize site-specific binary matrices
    sites = sorted(df['site'].unique())
    site_binary_matrices = {site: pd.DataFrame(0, index=frames, columns=base_pairs) for site in sites}

    print("\nBuilding site-encoded and site-specific matrices...")
    for frame in tqdm(frames, desc="Processing frames for matrices"):
        for base_pair in base_pairs:
            contact_sites = []
            # Check each site for this base_pair at this frame
            for site in sites:
                residue_pair_site = f"{base_pair}_site{site}"
                if residue_pair_site in distance_matrix.columns:
                    dist = distance_matrix.loc[frame, residue_pair_site]
                    if pd.notna(dist):
                        # Determine threshold based on mode
                        mode = base_pair.split('_')[-1]
                        threshold = 4.0 # Default (salt)
                        if mode == 'hbond': threshold = 3.5
                        elif mode == 'vdw': threshold = 3.8
                        
                        if dist <= threshold:
                            contact_sites.append(site)
                            # Record binary contact for this specific site
                            site_binary_matrices[site].loc[frame, base_pair] = 1
            
            # Assign site number to matrix for plotting. Prioritize lower site number if multiple contacts.
            if contact_sites:
                site_matrix.loc[frame, base_pair] = min(contact_sites) 
                # Could use a different value (e.g., 4) if len(contact_sites) > 1

    print(f"\nDebug - Site-encoded matrix (for plotting) information:")
    print(f"Shape: {site_matrix.shape} (frames × base residue pairs)")
    print(f"Base residue pairs: {site_matrix.columns.tolist()}")
    print(f"Frame range: {site_matrix.index.min()} to {site_matrix.index.max()}")
    print(f"Contact values distribution: {site_matrix.stack().value_counts().to_dict()}")

    for site in sites:
        print(f"\nDebug - Site {site} Binary Matrix information:")
        print(f"Shape: {site_binary_matrices[site].shape}")
        print(f"Number of contacts: {site_binary_matrices[site].values.sum()}")

    return switches, site_matrix, site_binary_matrices

# === NEW FUNCTION: Analyze Contact Co-occurrence ===
def analyze_contact_cooccurrence(site_binary_matrices):
    """
    Analyzes co-occurrence of contacts across different sites, focusing on Site 1's status.
    1. Calculates co-occurrence of Site 2/3 contacts when Site 1 IS contacting.
    2. Calculates co-occurrence of Site 2/3 contacts when Site 1 IS NOT contacting.

    Args:
        site_binary_matrices (dict): Dict of {site_num: binary_matrix} for each site.
                                    Assumes matrices have same index (frames) and columns (base_pairs).
                                    
    Returns:
        tuple: (results_present_df, results_absent_df)
               DataFrames containing the analysis results for both scenarios.
    """
    print("\n[AIM 2.5] Analyzing Contact Co-occurrence (Site 1 vs Site 2/3)...")

    sites = sorted(site_binary_matrices.keys())
    if 1 not in sites:
        print("Error: Site 1 matrix not found. Cannot perform co-occurrence analysis.")
        return None, None
        
    site1_matrix = site_binary_matrices[1]
    base_pairs = site1_matrix.columns
    n_frames = len(site1_matrix)
    all_frames = site1_matrix.index

    results_present = [] # Results when site 1 IS contacting
    results_absent = []  # Results when site 1 IS NOT contacting

    for pair in base_pairs:
        # Frames where site 1 has contact for this pair
        site1_contact_frames = site1_matrix.index[site1_matrix[pair] == 1]
        n_site1_contacts = len(site1_contact_frames)
        
        # Frames where site 1 DOES NOT have contact for this pair
        site1_no_contact_frames = site1_matrix.index[site1_matrix[pair] == 0]
        n_site1_no_contacts = len(site1_no_contact_frames)

        # --- Analysis 1: When Site 1 IS contacting ---
        if n_site1_contacts > 0:
            co_occurrence_present_site2 = 0
            co_occurrence_present_site3 = 0

            if 2 in sites:
                site2_matrix = site_binary_matrices[2]
                co_occurrence_present_site2 = site2_matrix.loc[site1_contact_frames, pair].sum()
                
            if 3 in sites:
                site3_matrix = site_binary_matrices[3]
                co_occurrence_present_site3 = site3_matrix.loc[site1_contact_frames, pair].sum()
            
            percent_present_site2 = (co_occurrence_present_site2 / n_site1_contacts) * 100
            percent_present_site3 = (co_occurrence_present_site3 / n_site1_contacts) * 100

            results_present.append({
                'residue_pair': pair,
                'site1_contact_frames': n_site1_contacts,
                'site2_contacts_during_site1': co_occurrence_present_site2,
                'site3_contacts_during_site1': co_occurrence_present_site3,
                'percent_site2_during_site1': percent_present_site2,
                'percent_site3_during_site1': percent_present_site3
            })

        # --- Analysis 2: When Site 1 IS NOT contacting ---
        if n_site1_no_contacts > 0:
            co_occurrence_absent_site2 = 0
            co_occurrence_absent_site3 = 0

            if 2 in sites:
                site2_matrix = site_binary_matrices[2]
                co_occurrence_absent_site2 = site2_matrix.loc[site1_no_contact_frames, pair].sum()
                
            if 3 in sites:
                site3_matrix = site_binary_matrices[3]
                co_occurrence_absent_site3 = site3_matrix.loc[site1_no_contact_frames, pair].sum()
                
            percent_absent_site2 = (co_occurrence_absent_site2 / n_site1_no_contacts) * 100
            percent_absent_site3 = (co_occurrence_absent_site3 / n_site1_no_contacts) * 100

            results_absent.append({
                'residue_pair': pair,
                'site1_no_contact_frames': n_site1_no_contacts,
                'site2_contacts_without_site1': co_occurrence_absent_site2,
                'site3_contacts_without_site1': co_occurrence_absent_site3,
                'percent_site2_without_site1': percent_absent_site2,
                'percent_site3_without_site1': percent_absent_site3
            })

    # --- Process Results for Site 1 Present ---
    results_present_df = pd.DataFrame() # Initialize empty dataframe
    if results_present:
        results_present_df = pd.DataFrame(results_present)
        results_present_df = results_present_df.sort_values(by='site1_contact_frames', ascending=False)

        print("\nCo-occurrence Summary (During Site 1 Contact):")
        print(results_present_df.to_string(index=False, 
                                 float_format="{:.1f}%".format, 
                                 columns=['residue_pair', 'site1_contact_frames', 'percent_site2_during_site1', 'percent_site3_during_site1']))

        output_csv_present = os.path.join(OUTPUT_FOLDER, "cooccurrence_site1_present.csv")
        results_present_df.to_csv(output_csv_present, index=False, float_format='%.3f')
        print(f"\nCo-occurrence analysis (Site 1 present) saved to {output_csv_present}")
    else:
        print("\nNo Site 1 contacts found for any pair. Cannot generate 'Site 1 Present' co-occurrence report.")

    # --- Process Results for Site 1 Absent ---
    results_absent_df = pd.DataFrame() # Initialize empty dataframe
    if results_absent:
        results_absent_df = pd.DataFrame(results_absent)
        results_absent_df = results_absent_df.sort_values(by='site1_no_contact_frames', ascending=False)

        print("\nCo-occurrence Summary (During Site 1 Absence):")
        print(results_absent_df.to_string(index=False, 
                                float_format="{:.1f}%".format, 
                                columns=['residue_pair', 'site1_no_contact_frames', 'percent_site2_without_site1', 'percent_site3_without_site1']))

        output_csv_absent = os.path.join(OUTPUT_FOLDER, "cooccurrence_site1_absent.csv")
        results_absent_df.to_csv(output_csv_absent, index=False, float_format='%.3f')
        print(f"\nCo-occurrence analysis (Site 1 absent) saved to {output_csv_absent}")
    else:
        print("\nNo frames without Site 1 contact found for any pair. Cannot generate 'Site 1 Absent' co-occurrence report.")
        
    return results_present_df, results_absent_df

# === NEW Plotting Functions: Co-occurrence by Site-Specific Interaction ===
def plot_cooccurrence_by_site_interaction_present(df_present, site2_base_pairs, site3_base_pairs):
    """Plots Site 2/3 contact % during Site 1 contact, ensuring all S2/S3 labels are shown."""
    # Generate the complete list of expected site-specific pairs and map to site number
    expected_pairs_dict = {}
    if site2_base_pairs:
        for p in site2_base_pairs:
            expected_pairs_dict[f"{p}_site2"] = {'percentage': 0.0, 'site': 2}
    if site3_base_pairs:
        for p in site3_base_pairs:
            expected_pairs_dict[f"{p}_site3"] = {'percentage': 0.0, 'site': 3}
            
    if not expected_pairs_dict:
        print("Skipping plot: No Site 2 or Site 3 interactions found in input data.")
        return

    # Populate percentages from the analysis results (df_present)
    if df_present is not None and not df_present.empty:
        for index, row in df_present.iterrows():
            base_pair = row['residue_pair']
            # Update Site 2 if applicable
            site2_label = f"{base_pair}_site2"
            if site2_label in expected_pairs_dict and 'percent_site2_during_site1' in row:
                expected_pairs_dict[site2_label]['percentage'] = row['percent_site2_during_site1']
            # Update Site 3 if applicable
            site3_label = f"{base_pair}_site3"
            if site3_label in expected_pairs_dict and 'percent_site3_during_site1' in row:
                expected_pairs_dict[site3_label]['percentage'] = row['percent_site3_during_site1']

    # Create DataFrame for plotting from the populated dictionary
    plot_data = []
    for label, data in expected_pairs_dict.items():
        plot_data.append({
            'site_specific_pair': label,
            'percentage': data['percentage'],
            'site': data['site']
        })
        
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values('site_specific_pair')

    # Map colors based on the site derived from the label
    colors = plot_df['site'].map({2: 'blue', 3: 'orange'})

    # --- Debug: Print head of DataFrame before plotting ---
    print("\nDebug: Plotting data for plot_cooccurrence_by_site_interaction_present:")
    print(plot_df.head().to_string())
    # ---

    plt.figure(figsize=(18, 8))
    bars = plt.bar(plot_df['site_specific_pair'], plot_df['percentage'], color=colors, alpha=0.7)

    plt.xlabel('Site-Specific Residue Pair')
    plt.ylabel('Percentage of Site 1 Contact Frames (%)')
    plt.title('Site 2 & 3 Contact Occurrence During Site 1 Contact (by Site-Specific Interaction)')
    plt.xticks(ticks=range(len(plot_df)), labels=plot_df['site_specific_pair'], rotation=75, ha='right')
    plt.ylim(0, 110) # Increase ylim slightly for text labels
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Create custom legend
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', alpha=0.7, label='Site 2 Interaction'),
        Patch(facecolor='orange', edgecolor='orange', alpha=0.7, label='Site 3 Interaction')
    ]
    plt.legend(handles=legend_elements, title="Interaction Site")
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0: # Only label non-zero bars
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%', # Format to one decimal place
                     ha='center', va='bottom', fontsize=8) # Adjust fontsize as needed
                    
    plt.tight_layout()

    output_plot = os.path.join(OUTPUT_FOLDER, "cooccurrence_bysite_plot_site1_present.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Co-occurrence plot by site (Site 1 present) saved to {output_plot}")

def plot_cooccurrence_by_site_interaction_absent(df_absent, site2_base_pairs, site3_base_pairs):
    """Plots Site 2/3 contact % during Site 1 absence, ensuring all S2/S3 labels are shown."""
    # Generate the complete list of expected site-specific pairs and map to site number
    expected_pairs_dict = {}
    if site2_base_pairs:
        for p in site2_base_pairs:
            expected_pairs_dict[f"{p}_site2"] = {'percentage': 0.0, 'site': 2}
    if site3_base_pairs:
        for p in site3_base_pairs:
            expected_pairs_dict[f"{p}_site3"] = {'percentage': 0.0, 'site': 3}
            
    if not expected_pairs_dict:
        print("Skipping plot: No Site 2 or Site 3 interactions found in input data.")
        return

    # Populate percentages from the analysis results (df_absent)
    if df_absent is not None and not df_absent.empty:
        for index, row in df_absent.iterrows():
            base_pair = row['residue_pair']
            # Update Site 2 if applicable
            site2_label = f"{base_pair}_site2"
            if site2_label in expected_pairs_dict and 'percent_site2_without_site1' in row:
                expected_pairs_dict[site2_label]['percentage'] = row['percent_site2_without_site1']
            # Update Site 3 if applicable
            site3_label = f"{base_pair}_site3"
            if site3_label in expected_pairs_dict and 'percent_site3_without_site1' in row:
                expected_pairs_dict[site3_label]['percentage'] = row['percent_site3_without_site1']

    # Create DataFrame for plotting from the populated dictionary
    plot_data = []
    for label, data in expected_pairs_dict.items():
        plot_data.append({
            'site_specific_pair': label,
            'percentage': data['percentage'],
            'site': data['site']
        })
        
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values('site_specific_pair')

    # Map colors based on the site derived from the label
    colors = plot_df['site'].map({2: 'blue', 3: 'orange'})

    # --- Debug: Print head of DataFrame before plotting ---
    print("\nDebug: Plotting data for plot_cooccurrence_by_site_interaction_absent:")
    print(plot_df.head().to_string())
    # ---

    plt.figure(figsize=(18, 8))
    bars = plt.bar(plot_df['site_specific_pair'], plot_df['percentage'], color=colors, alpha=0.7)

    plt.xlabel('Site-Specific Residue Pair')
    plt.ylabel('Percentage of Site 1 Absence Frames (%)')
    plt.title('Site 2 & 3 Contact Occurrence During Site 1 Absence (by Site-Specific Interaction)')
    plt.xticks(ticks=range(len(plot_df)), labels=plot_df['site_specific_pair'], rotation=75, ha='right')
    plt.ylim(0, 110) # Increase ylim slightly for text labels
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', alpha=0.7, label='Site 2 Interaction'),
        Patch(facecolor='orange', edgecolor='orange', alpha=0.7, label='Site 3 Interaction')
    ]
    plt.legend(handles=legend_elements, title="Interaction Site")
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0: # Only label non-zero bars
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%', # Format to one decimal place
                     ha='center', va='bottom', fontsize=8) # Adjust fontsize as needed
                    
    plt.tight_layout()

    output_plot = os.path.join(OUTPUT_FOLDER, "cooccurrence_bysite_plot_site1_absent.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Co-occurrence plot by site (Site 1 absent) saved to {output_plot}")

# === FUNCTION 4: AIM 3 – Rebinding Events ===
def count_rebinding_events(df):
    """
    Counts how many times each Arg59–X interaction goes from 'off' to 'on' (rebinding).
    Uses different thresholds for different interaction types:
    - Hydrogen bonds: ≤ 3.5Å
    - van der Waals: ≤ 3.8Å
    - Salt bridges: ≤ 4.0Å
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Mode is already extracted
    # df['mode'] = df['residue_pair'].str.split('_').str[-2]
    
    rebinding_counts = defaultdict(int)

    # Group by the full residue_pair (which includes site)
    for pair, subdf in df.groupby('residue_pair'):
        subdf = subdf.sort_values('frame')
        
        # Get the appropriate threshold based on interaction type (mode)
        mode = pair.split('_')[-2] # Mode is second to last
        if mode == 'hbond':
            pair_threshold = 3.5
        elif mode == 'vdw':
            pair_threshold = 3.8
        else: # Assume salt bridge otherwise
            pair_threshold = 4.0
            
        prev = False
        for dist in subdf['distance']:
            curr = dist <= pair_threshold
            if not prev and curr:
                rebinding_counts[pair] += 1
            prev = curr

    rebinding_df = pd.DataFrame.from_dict(rebinding_counts, orient='index', columns=['rebinding_count'])
    rebinding_df = rebinding_df.sort_values('rebinding_count', ascending=False).reset_index()
    rebinding_df.columns = ['residue_pair', 'rebinding_count']
    
    # Add back site, mode, residue2 for plotting convenience
    rebinding_df['site'] = rebinding_df['residue_pair'].str.split('_').str[-1].str.replace('site', '').astype(int)
    rebinding_df['mode'] = rebinding_df['residue_pair'].str.split('_').str[-2]
    rebinding_df['residue2'] = rebinding_df['residue_pair'].str.split('_').str[1]
    
    return rebinding_df

def plot_contact_frequencies(freq_df):
    """Create a bar plot of contact frequencies, labelled by site"""
    plt.figure(figsize=(18, 8)) # Wider figure for more labels
    
    # Mode and residue2 are already in freq_df
    # Extract site number for labeling
    # freq_df['site'] = freq_df['residue_pair'].str.split('_').str[-1].str.replace('site', '')
    # freq_df['mode'] = freq_df['residue_pair'].str.split('_').str[-2]
    # freq_df['residue2'] = freq_df['residue_pair'].str.split('_').str[1]
    
    # Create the label combining residue, mode, and site
    freq_df['label'] = freq_df['residue_pair'] # Use the full unique name as label initially
                                               # Or format nicely: freq_df.apply(lambda row: f"Arg59_{row['residue2']}_site{row['site']}", axis=1)
    freq_df = freq_df.sort_values(by=['residue2', 'mode', 'site']) # Sort for consistent plotting order

    # Check for entries without proper mode suffix
    valid_modes = {'salt', 'hbond', 'vdw'}
    invalid_entries = freq_df[~freq_df['mode'].isin(valid_modes)]
    if not invalid_entries.empty:
        print("\nWarning: Found entries without proper interaction mode suffix (_salt, _hbond, or _vdw) preceding the site suffix:")
        for _, row in invalid_entries.iterrows():
            print(f"  - {row['residue_pair']}")
        print("Fix CSV file naming convention (e.g., Arg59_Glu76_salt.csv).")
        # Consider raising error if strict checking is needed
        # raise ValueError("Invalid interaction mode detected in residue pair names.")
        
    # Calculate total contact frequency and no-contact frequency
    # This definition might need refinement if contacts can be simultaneous across sites
    # For now, sum all frequencies as before.
    total_contact_freq = freq_df['contact_frequency'].sum()
    no_contact_freq = max(0, 1.0 - total_contact_freq) # Ensure non-negative
    
    # Create a DataFrame row for no contacts
    no_contact_row = pd.DataFrame({
        'residue_pair': ['No_Contacts'],
        'contact_frequency': [no_contact_freq],
        'mode': ['none'],
        'residue2': ['No Contacts'],
        'site': [0],
        'label': ['No Contacts']
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
    
    # Map colors based on mode
    plot_df['color'] = plot_df['mode'].map(mode_colors)
    
    # Plot bars using the calculated color
    plt.bar(plot_df['label'], plot_df['contact_frequency'], 
            color=plot_df['color'], alpha=0.7)

    # Add horizontal line at 1.0
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.text(0, 1.02, 'Total possible contacts = 1.0 (may exceed 1 if simultaneous contacts exist)', color='gray', fontsize=10)
    
    plt.xticks(rotation=60, ha='right') # Increase rotation for longer labels
    plt.ylabel('Contact Frequency')
    plt.title('Contact Frequencies of Arg59 (Color by Mode, Labeled by Site)')
    
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
              bbox_to_anchor=(1.15, 1)) # Adjust legend position if needed
    
    # Set y-axis limits
    plt.ylim(0, max(1.1, plot_df['contact_frequency'].max() * 1.1)) # Adjust upper limit based on data
    
    plt.tight_layout()
    
    output_plot = os.path.join(OUTPUT_FOLDER, "contact_frequencies_plot_all_sites.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Contact frequencies plot saved to {output_plot}")
    
    # Print the frequencies including no contacts
    print("\nContact Frequencies Summary (per site):")
    for _, row in plot_df.iterrows():
        print(f"{row['label']}: {row['contact_frequency']:.3f}")
    print(f"Total (check interpretation if > 1.0): {plot_df['contact_frequency'].sum():.3f}")

def plot_rebinding_counts(rebinding_df):
    """Create two versions of the rebinding counts plot - labelled by site, colored by mode"""
    
    # Mode, residue2, site are already in rebinding_df
    # Create the label combining residue, mode, and site
    rebinding_df['label'] = rebinding_df['residue_pair'] # Use the full unique name
    rebinding_df = rebinding_df.sort_values(by=['residue2', 'mode', 'site']) # Sort for consistent plotting order

    # Create color mapping for different modes
    mode_colors = {
        'salt': '#D55E00',  # Orange-red
        'hbond': '#56B4E9', # Light blue
        'vdw': '#006837'    # Dark green
    }
    # Map colors based on mode
    rebinding_df['color'] = rebinding_df['mode'].map(mode_colors)

    # Update legend labels
    legend_labels = {
        'salt': 'Salt Bridge',
        'hbond': 'Hydrogen Bond',
        'vdw': 'van der Waals'
    }
    # Create legend handles for all potential modes
    legend_handles = [plt.Rectangle((0,0),1,1, color=mode_colors[mode], alpha=0.7) 
                     for mode in mode_colors.keys()]
    legend_texts = [legend_labels[mode] for mode in mode_colors.keys()]

    # --- Version 1: With grid lines and count numbers ---
    plt.figure(figsize=(18, 8)) # Wider figure
    
    bars = plt.bar(rebinding_df['label'], rebinding_df['rebinding_count'], 
                   color=rebinding_df['color'], alpha=0.7)
    
    # Add count numbers on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.xticks(rotation=60, ha='right') # Increase rotation
    plt.ylabel('Number of Rebinding Events')
    plt.title('Rebinding Events (Color by Mode, Labeled by Site - with grid)')
    
    # Place legend in the upper right corner of the plot
    plt.legend(legend_handles, legend_texts,
              title='Interaction Mode',
              loc='upper right',
              # bbox_to_anchor=(0.98, 0.98), # May need adjustment
              frameon=True,
              framealpha=0.9)
    
    # Add grid lines
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, rebinding_df['rebinding_count'].max() * 1.15) # Adjust ylim for text labels
    
    plt.tight_layout()
    
    output_plot_grid = os.path.join(OUTPUT_FOLDER, "rebinding_counts_plot_all_sites_with_grid.png")
    plt.savefig(output_plot_grid, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rebinding counts plot with grid lines saved to {output_plot_grid}")
    
    # --- Version 2: Without grid lines ---
    plt.figure(figsize=(18, 8)) # Wider figure
    
    plt.bar(rebinding_df['label'], rebinding_df['rebinding_count'], 
            color=rebinding_df['color'], alpha=0.7)
    
    plt.xticks(rotation=60, ha='right') # Increase rotation
    plt.ylabel('Number of Rebinding Events')
    plt.title('Rebinding Events (Color by Mode, Labeled by Site - without grid)')
    
    # Create legend
    plt.legend(legend_handles, legend_texts,
              title='Interaction Mode',
              loc='upper right',
              # bbox_to_anchor=(0.98, 0.98), # May need adjustment
              frameon=True,
              framealpha=0.9)
    
    # No grid lines in this version
    plt.grid(False)
    plt.ylim(0, rebinding_df['rebinding_count'].max() * 1.1) # Adjust ylim
    
    plt.tight_layout()
    
    output_plot_no_grid = os.path.join(OUTPUT_FOLDER, "rebinding_counts_plot_all_sites_without_grid.png")
    plt.savefig(output_plot_no_grid, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rebinding counts plot without grid lines saved to {output_plot_no_grid}")

def plot_contact_matrix(site_contact_matrix):
    """Create a heatmap of the contact matrix, colored by site"""
    plt.figure(figsize=(16, 14))  # Adjust size as needed
    
    # Define colors for sites and no contact
    # 0: No Contact, 1: Site 1, 2: Site 2, 3: Site 3
    # Add more if needed (e.g., for multiple simultaneous contacts)
    site_colors = {
        0: 'white', 
        1: 'red', 
        2: 'blue', 
        3: 'orange' 
        # 4: 'purple' # Example if using 4 for multiple contacts
    }
    
    # Create a colormap from the dictionary
    cmap_list = [site_colors[i] for i in sorted(site_colors.keys())]
    cmap = plt.cm.colors.ListedColormap(cmap_list)
    
    # Create boundaries for the colormap
    bounds = list(sorted(site_colors.keys())) + [max(site_colors.keys()) + 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Create custom legend patches
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='No Contact'),
        Patch(facecolor='red', edgecolor='black', label='Site 1 Contact'),
        Patch(facecolor='blue', edgecolor='black', label='Site 2 Contact'),
        Patch(facecolor='orange', edgecolor='black', label='Site 3 Contact'),
        # Add entry for multiple contacts if applicable
        # Patch(facecolor='purple', edgecolor='black', label='Multiple Sites Contact') 
    ]
    
    # Create the heatmap
    ax = plt.gca()
    sns.heatmap(site_contact_matrix, 
                cmap=cmap,
                norm=norm,
                cbar=False,  # No numerical colorbar
                xticklabels=site_contact_matrix.columns, 
                yticklabels=True) # Show frame numbers
    
    # Set y-axis (frame) tick intervals
    n_frames = len(site_contact_matrix)
    tick_interval = 50
    tick_positions_y = np.arange(0, n_frames, tick_interval)
    plt.yticks(tick_positions_y, tick_positions_y)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=60, ha='right')
    
    plt.xlabel('Residue Pair')
    plt.ylabel('Frame Number')
    plt.title('Contact Matrix (Colored by Site)')
    
    # Add custom legend below the plot
    ax.legend(handles=legend_elements, 
              bbox_to_anchor=(0.5, -0.25), # Adjust position below plot
              loc='upper center',
              frameon=True,
              edgecolor='black',
              fontsize=12,
              ncol=len(legend_elements)) # Adjust columns based on number of elements
    
    # Adjust layout
    plt.tight_layout()
    
    output_plot = os.path.join(OUTPUT_FOLDER, "contact_matrix_heatmap_all_sites.png")
    # Increase bottom margin to make space for legend
    plt.savefig(output_plot, dpi=300, bbox_inches='tight', pad_inches=0.5) 
    plt.close()
    print(f"Site-colored contact matrix heatmap saved to {output_plot}")

# === FUNCTION 5: Find no contact frames (adapting for site-encoded matrix) ===
def find_no_contact_frames(site_contact_matrix):
    """
    Finds frames where there are no contacts across any site/pair.
    Analyzes consecutive runs of these no-contact frames.
    
    Parameters:
    -----------
    site_contact_matrix : pandas DataFrame or numpy array
        The site-encoded contact matrix (0 = no contact, >0 = contact on some site).
    
    Returns:
    --------
    tuple
        - no_contact_frames (np.array): Frame indices where no contacts were observed.
        - no_contact_periods (list): List of (start_frame, end_frame, duration) 
                                     for consecutive no-contact periods, sorted by duration desc.
    """
    # Sum across columns (base residue pairs) for each frame. If sum is 0, no contact on any site.
    contact_sums = site_contact_matrix.sum(axis=1)
    # Find indices where sum is 0 (no contacts)
    no_contact_frame_indices = site_contact_matrix.index[contact_sums == 0]
    no_contact_frames = np.array(no_contact_frame_indices) # Convert to numpy array if needed for consistency
    
    # Analyze gaps between no-contact frames
    no_contact_periods = []
    if len(no_contact_frames) > 0:
        # Find consecutive runs in the no_contact_frames array
        if len(no_contact_frames) == 1:
            # Only one frame with no contact
            start_frame = no_contact_frames[0]
            no_contact_periods.append((start_frame, start_frame, 1))
        else:
            diffs = np.diff(no_contact_frames)
            split_indices = np.where(diffs != 1)[0] + 1
            runs = np.split(no_contact_frames, split_indices)
            
            for run in runs:
                if len(run) > 0:
                    start_frame = run[0]
                    end_frame = run[-1]
                    duration = len(run)
                    no_contact_periods.append((start_frame, end_frame, duration))
        
        # Sort periods by duration in descending order
        no_contact_periods.sort(key=lambda x: x[2], reverse=True)
    
    return no_contact_frames, no_contact_periods

# === MAIN FUNCTION: Run all three aims ===
def main():
    print("Delete 2d rmsd matrix dataframe if you want to recompute it.")
    
    # Define CSV folders
    csv_folders = [
        os.path.join(root_path, "bn_bs_basin_2/data/site_1"),
        os.path.join(root_path, "bn_bs_basin_2/data/site_2"),
        os.path.join(root_path, "bn_bs_basin_2/data/site_3")
    ]

    # Get number of frames from DCD file once
    print("\nLoading trajectory information...")
    u = mda.Universe(PDB_PATH, DCD_PATH)
    n_frames = len(u.trajectory)
    print(f"Detected {n_frames} frames in DCD file")
    
    print("\nLoading distance data...")
    df = load_distance_data(csv_folders, n_frames) # Pass list of folders

    print("\n[AIM 1] Calculating contact frequencies...")
    freq_df = calculate_contact_frequencies(df)
    print(freq_df)
    output_freq = os.path.join(OUTPUT_FOLDER, "contact_frequencies.csv")
    freq_df.to_csv(output_freq, index=False)
    print(f"Contact frequencies saved to {output_freq}")
    plot_contact_frequencies(freq_df)

    print("\n[AIM 2] Calculating switching events and contact matrix...")
    # Returns switch count (binary logic) and site-encoded matrix for plotting
    switch_count, site_contact_matrix, site_binary_matrices = calculate_switching_events(df)
    print(f"Total switching events (based on binary contact changes per site): {switch_count}")
    
    # Save the site-encoded matrix
    output_site_matrix = os.path.join(OUTPUT_FOLDER, "contact_matrix_site_encoded.csv")
    site_contact_matrix.to_csv(output_site_matrix)
    print(f"Site-encoded contact matrix saved to {output_site_matrix}")
    
    # Plot the site-encoded matrix
    plot_contact_matrix(site_contact_matrix)
    
    # Analyze contact co-occurrence using the binary matrices
    results_present_df, results_absent_df = analyze_contact_cooccurrence(site_binary_matrices)
    
    # Get lists of base pairs actually present for sites 2 and 3
    site2_base_pairs = site_binary_matrices[2].columns[site_binary_matrices[2].sum() > 0].tolist() if 2 in site_binary_matrices else []
    site3_base_pairs = site_binary_matrices[3].columns[site_binary_matrices[3].sum() > 0].tolist() if 3 in site_binary_matrices else []
    print(f"\nDebug: Base pairs found for Site 2: {site2_base_pairs}")
    print(f"Debug: Base pairs found for Site 3: {site3_base_pairs}")
    
    # Plot the co-occurrence results
    # Original plots (grouped bars per base interaction)
    # plot_cooccurrence_site1_present(results_present_df)
    # plot_cooccurrence_site1_absent(results_absent_df)
    
    # New plots (bars per site-specific interaction)
    plot_cooccurrence_by_site_interaction_present(results_present_df, site2_base_pairs, site3_base_pairs)
    plot_cooccurrence_by_site_interaction_absent(results_absent_df, site2_base_pairs, site3_base_pairs)
    
    # Find frames with no contacts using the site-encoded matrix
    no_contact_frames, no_contact_periods = find_no_contact_frames(site_contact_matrix)
    print(f"\nFound {len(no_contact_frames)} frames with no contacts across any site:")
    if len(no_contact_frames) > 0:
        print(f"Frame numbers: {no_contact_frames.tolist()}")
        # Save the frame numbers to a file
        output_frames = os.path.join(OUTPUT_FOLDER, "no_contact_frames.txt")
        np.savetxt(output_frames, no_contact_frames, fmt='%d')
        print(f"No-contact frame numbers saved to {output_frames}")
        
        # Calculate percentage of frames with no contacts
        percent_no_contacts = (len(no_contact_frames) / len(site_contact_matrix)) * 100
        print(f"Percentage of frames with no contacts: {percent_no_contacts:.2f}%")
        
        # Analyze gaps between no-contact frames
        if no_contact_periods:
            print("\nConsecutive Periods Without Any Contact:")
            print(f"Total number of no-contact periods: {len(no_contact_periods)}")
            
            # Basic gap statistics
            durations = [p[2] for p in no_contact_periods]
            print(f"Shortest no-contact period: {min(durations)} frames")
            print(f"Longest no-contact period: {max(durations)} frames")
            print(f"Average no-contact period: {np.mean(durations):.2f} frames")
            print(f"Median no-contact period: {np.median(durations):.0f} frames")
            
            # Report the top 3 longest periods
            print("\nTop 3 longest consecutive periods without any contact:")
            for i, (start, end, duration) in enumerate(no_contact_periods[:3], 1):
                print(f"{i}. Frames {start} - {end} (duration: {duration} frames)")
            
            # Save gap analysis to file
            output_periods = os.path.join(OUTPUT_FOLDER, "no_contact_periods.csv")
            no_contact_periods_df = pd.DataFrame(no_contact_periods, columns=['start_frame', 'end_frame', 'duration'])
            no_contact_periods_df.to_csv(output_periods, index=False)
            print(f"\nDetails of all no-contact periods saved to {output_periods}")
        else:
            # This case should ideally not happen if len(no_contact_frames) > 0
            print("No consecutive no-contact periods found (check logic if this message appears).") 
    else:
        print("No frames without any contacts were found.")

    print("\n[AIM 3] Counting rebinding events...")
    rebinding_df = count_rebinding_events(df)
    print(rebinding_df[['residue_pair', 'rebinding_count']]) # Print relevant columns
    output_rebind = os.path.join(OUTPUT_FOLDER, "rebinding_counts_by_site.csv")
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
