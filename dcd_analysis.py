#!/scratch/users/lm18di/anaconda3/envs/md/bin/python

# usage: dcd_analysis.py hosts/hostname/system_name/

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import MDAnalysis as mda
from sklearn.cluster import KMeans
from MDAnalysis.analysis import align
import time
import psutil

def setup_logging(logs_output_dir, system):
    """Set up logging to both console and file"""
    log_file = os.path.join(logs_output_dir, f"{system}_dcd_analysis.log")
    
    # Create logger
    logger = logging.getLogger('dcd_analysis')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def is_empty_dcd(dcd_path):                                               
    """Check if a DCD file is empty or corrupted"""          
    try:                                                                     
        file_size = os.path.getsize(dcd_path)
        if file_size < 100:
            return True               
                               
        u = mda.Universe(dcd_path)                     
        if len(u.trajectory) == 0:
            return True                                                                              
                               
        u.trajectory[0]                  
        return False                     
    except Exception as e:
        print(f"Warning: Error reading DCD file {dcd_path}: {str(e)}")
        return True
    
def get_restart_folder_depth(path):
    """Calculate how many restart folders deep this path is"""
    parts = os.path.normpath(path).split(os.sep)
    restart_count = parts.count('restart')
    return restart_count

def find_all_dcd_files(base_dir):
    """Find all valid DCD files in the base directory and its subdirectories, excluding minima_* directories"""
    print("Searching for DCD files (this may take a while for large directory trees)...")
    sys.stdout.flush()
    
    dcd_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip minima_* directories
        dirs[:] = [d for d in dirs if not d.startswith('minima_')]
        
        for file in files:
            if file.endswith('.dcd'):
                dcd_path = os.path.join(root, file)
                if 'minima_' in dcd_path:  # Additional check for nested paths
                    continue
                if is_empty_dcd(dcd_path):
                    print(f"Skipping empty/corrupt DCD file: {dcd_path}")
                    continue
                dcd_files.append(dcd_path)
                print(f"Found valid DCD file: {dcd_path}")
                sys.stdout.flush()
    
    # Sort by restart folder depth and path length
    dcd_files.sort(key=lambda path: (get_restart_folder_depth(path), path.count(os.sep)))
    return dcd_files

def find_psf_or_pdb_file(base_dir):
    """Find a PSF file in the base directory or restart subfolders. If no PSF is found, look for a PDB file."""
    # First look for PSF in the base directory
    for file in os.listdir(base_dir):
        if file.endswith('.psf'):
            print("Found PSF file:", os.path.join(base_dir, file))
            return os.path.join(base_dir, file)
    
    # If no PSF found in the base directory, look in first-level restart folder
    restart_dir = os.path.join(base_dir, 'restart')
    if os.path.exists(restart_dir) and os.path.isdir(restart_dir):
        for file in os.listdir(restart_dir):
            if file.endswith('.psf'):
                print("Found PSF file in restart folder:", os.path.join(restart_dir, file))
                return os.path.join(restart_dir, file)
    
    # Still nothing? Search the whole tree for PSF
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.psf'):
                print("Found PSF file in subdirectory:", os.path.join(root, file))
                return os.path.join(root, file)
    
    # If no PSF found, look for PDB files in the same order
    print("No PSF file found, looking for PDB file...")
    
    # Look in base directory
    for file in os.listdir(base_dir):
        if file.endswith('.pdb'):
            print("Found PDB file:", os.path.join(base_dir, file))
            return os.path.join(base_dir, file)
    
    # Look in restart folder
    if os.path.exists(restart_dir) and os.path.isdir(restart_dir):
        for file in os.listdir(restart_dir):
            if file.endswith('.pdb'):
                print("Found PDB file in restart folder:", os.path.join(restart_dir, file))
                return os.path.join(restart_dir, file)
    
    # Search the whole tree for PDB
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pdb'):
                print("Found PDB file in subdirectory:", os.path.join(root, file))
                return os.path.join(root, file)
    
    print("Warning: No PSF or PDB file found!")
    return None

def sample_frames_smart(frame_df, max_frames, rmsd_col='rmsd'):
    """
    Sample frames while preserving RMSD distribution.
    Args:
        frame_df: DataFrame containing frame numbers and RMSD values
        max_frames: Maximum number of frames to sample
        rmsd_col: Name of the RMSD column (default: 'rmsd')
    Returns:
        List of sampled frame numbers
    """
    if len(frame_df) <= max_frames:
        return frame_df['frame_number'].tolist()
    
    # Create bins based on RMSD distribution
    n_bins = min(50, max_frames // 20)  # Reasonable number of bins
    rmsd_values = frame_df[rmsd_col].values
    hist, bin_edges = np.histogram(rmsd_values, bins=n_bins)
    
    # Calculate how many samples to take from each bin to preserve distribution
    total_frames = len(frame_df)
    samples_per_bin = np.round((hist / total_frames) * max_frames).astype(int)
    
    # Ensure we don't exceed max_frames due to rounding
    while samples_per_bin.sum() > max_frames:
        idx = np.argmax(samples_per_bin)
        samples_per_bin[idx] -= 1
    
    # Sample frames from each bin
    sampled_frames = []
    for i in range(len(bin_edges) - 1):
        bin_mask = (rmsd_values >= bin_edges[i]) & (rmsd_values < bin_edges[i+1])
        bin_frames = frame_df[bin_mask]
        
        if len(bin_frames) > 0:
            n_samples = min(samples_per_bin[i], len(bin_frames))
            if n_samples > 0:
                # Stratified sampling within bin
                step = len(bin_frames) // n_samples
                indices = np.linspace(0, len(bin_frames)-1, n_samples, dtype=int)
                sampled_frames.extend(bin_frames.iloc[indices]['frame_number'].tolist())
    
    return sorted(sampled_frames)

def write_frames_to_dcd(universe, frame_numbers, output_file):
    """
    Efficiently write selected frames to a DCD file.
    Includes progress reporting and memory usage monitoring.
    """
    # Get total number of frames in trajectory
    total_frames = len(universe.trajectory)
    
    # Filter frame numbers to ensure they're within bounds
    valid_frames = [f for f in frame_numbers if 0 <= f < total_frames]
    
    if len(valid_frames) < len(frame_numbers):
        print(f"\nWarning: {len(frame_numbers) - len(valid_frames)} frames were outside trajectory bounds")
        print(f"Original frame range: {min(frame_numbers)} to {max(frame_numbers)}")
        print(f"Valid frame range: {min(valid_frames)} to {max(valid_frames)}")
        print(f"Total frames in trajectory: {total_frames}")
    
    if not valid_frames:
        raise ValueError("No valid frames found within trajectory bounds!")
    
    # Create a temporary universe with just the frames we want
    print(f"\nWriting {len(valid_frames)} frames to DCD file...")
    
    # Monitor memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Setup progress tracking
    start_time = time.time()
    last_progress_time = start_time
    update_interval = 500  # Update progress every 500 frames
    
    with mda.Writer(output_file, n_atoms=universe.atoms.n_atoms) as w:
        for i, frame_num in enumerate(valid_frames):
            try:
                universe.trajectory[frame_num]
                w.write(universe.atoms)
                
                # Report progress periodically
                if (i + 1) % update_interval == 0 or i == len(valid_frames) - 1:
                    current_time = time.time()
                    if current_time - last_progress_time >= 10:  # Don't update too frequently
                        elapsed = current_time - start_time
                        progress = (i + 1) / len(valid_frames) * 100
                        current_memory = process.memory_info().rss / 1024 / 1024
                        
                        print(f"Progress: {progress:.1f}% ({i+1}/{len(valid_frames)} frames)")
                        print(f"Elapsed time: {elapsed:.1f} seconds")
                        print(f"Current memory: {current_memory:.1f} MB (Delta: {current_memory - initial_memory:.1f} MB)")
                        
                        # Calculate ETA
                        if progress > 0:
                            eta = (elapsed / progress) * (100 - progress)
                            print(f"Estimated time remaining: {eta:.1f} seconds")
                        
                        last_progress_time = current_time
                        
            except (IndexError, StopIteration) as e:
                print(f"Error reading frame {frame_num}: {str(e)}")
                continue
    
    # Report final stats
    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"\nDCD file created successfully at: {output_file}")
    print(f"Total time: {end_time - start_time:.1f} seconds")
    print(f"Final memory usage: {final_memory:.1f} MB (Delta: {final_memory - initial_memory:.1f} MB)")

def create_rmsd_matrix(u, frame_numbers, output_path, minima_number, dvdl_df, logger):
    """Create a pairwise RMSD matrix for given frames and save as a plot.
    Uses only carbon alpha backbone atoms for more efficient calculation.
    Samples 1000 frames using RMSD-aware clustering."""
    start_time = time.time()
    
    # Select only CA atoms for more efficient RMSD calculation
    ca_atoms = u.select_atoms("name CA")
    logger.info(f"Using {len(ca_atoms)} carbon alpha atoms for RMSD calculation")
    
    # Get the frame DataFrame with RMSD values
    frame_df = dvdl_df[dvdl_df['frame_number'].isin(frame_numbers)].copy()
    
    # Sample exactly 1000 frames using RMSD-aware sampling
    max_frames_to_plot = 1000  # Set to exactly 1000 frames
    if len(frame_numbers) <= max_frames_to_plot:
        logger.info(f"\nNumber of frames ({len(frame_numbers)}) is within limit. Using all frames for RMSD matrix.")
        sampled_frames = frame_numbers
    else:
        logger.info(f"\nLarge number of frames ({len(frame_numbers)}) exceeds limit of {max_frames_to_plot}.")
        logger.info("Using RMSD-aware sampling to select 1000 representative frames...")
        sampled_frames = sample_frames_smart(frame_df, max_frames_to_plot, rmsd_col='rmsd')
        
        # Print sampling statistics
        original_rmsd = frame_df['rmsd']
        sampled_frames_df = frame_df[frame_df['frame_number'].isin(sampled_frames)]
        sampled_rmsd = sampled_frames_df['rmsd']
        
        logger.info("\nSampling Statistics:")
        logger.info(f"Original RMSD range: {original_rmsd.min():.3f} to {original_rmsd.max():.3f}")
        logger.info(f"Sampled RMSD range: {sampled_rmsd.min():.3f} to {sampled_rmsd.max():.3f}")
        logger.info(f"Original RMSD mean ± std: {original_rmsd.mean():.3f} ± {original_rmsd.std():.3f}")
        logger.info(f"Sampled RMSD mean ± std: {sampled_rmsd.mean():.3f} ± {sampled_rmsd.std():.3f}")
    
    # Convert frame numbers to integers and ensure they're valid
    sampled_frames = [int(f) for f in sampled_frames if not pd.isna(f)]
    n_frames = len(sampled_frames)
    logger.info(f"\nCalculating {n_frames}x{n_frames} RMSD matrix...")
    logger.info(f"Number of CA atoms: {len(ca_atoms)}")
    logger.info(f"Estimated number of calculations: {n_frames * (n_frames + 1) // 2}")
    
    rmsd_matrix = np.zeros((n_frames, n_frames))
    
    # Calculate RMSD between each pair of frames
    logger.info(f"Calculating RMSD matrix for {n_frames} frames using CA atoms only...")
    last_progress_time = time.time()
    for i, frame1 in enumerate(sampled_frames):
        try:
            u.trajectory[frame1]
            ref_coords = ca_atoms.positions.copy()
            for j, frame2 in enumerate(sampled_frames):
                if i <= j:  # Only calculate upper triangle
                    u.trajectory[frame2]
                    rmsd = mda.analysis.rms.rmsd(ca_atoms.positions, ref_coords, center=True, superposition=True)
                    rmsd_matrix[i, j] = rmsd
                    rmsd_matrix[j, i] = rmsd  # Symmetric matrix
            
            # Print progress more frequently
            current_time = time.time()
            if current_time - last_progress_time >= 60:  # Update every minute
                elapsed = current_time - start_time
                progress = (i + 1) / n_frames * 100
                estimated_total = elapsed / progress * 100 if progress > 0 else float('inf')
                remaining = estimated_total - elapsed
                
                logger.info(f"Progress: {progress:.1f}% ({i+1}/{n_frames} frames)")
                logger.info(f"Elapsed time: {elapsed/60:.1f} minutes")
                logger.info(f"Estimated remaining time: {remaining/60:.1f} minutes")
                last_progress_time = current_time
                
        except (IndexError, TypeError) as e:
            logger.warning(f"Error processing frame {frame1}: {str(e)}")
            continue
    
    # Create the plot - simplified to only show RMSD matrix
    logger.info("Creating RMSD matrix plot...")
    plt.figure(figsize=(10, 8))
    
    # Main RMSD matrix plot
    im = plt.imshow(rmsd_matrix, cmap='viridis', aspect='equal')
    plt.colorbar(im, label='RMSD (Å)')
    
    # Add frame numbers to axes, but limit the number of labels
    max_ticks = 20
    tick_step = max(1, n_frames // max_ticks)
    tick_positions = range(0, n_frames, tick_step)
    tick_labels = [f'Frame {sampled_frames[i]}' for i in tick_positions]
    
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    plt.yticks(tick_positions, tick_labels)
    
    plt.title(f'Pairwise RMSD Analysis - Minima {minima_number}\n'
             f'Using {len(sampled_frames)} frames from {len(frame_numbers)} total frames in minima\n'
             f'(Carbon Alpha RMSD, range: {frame_df["rmsd"].min():.2f} - {frame_df["rmsd"].max():.2f} Å)', 
             fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    total_time = time.time() - start_time
    logger.info(f"RMSD matrix plot saved to {output_path}")
    logger.info(f"Total time for RMSD matrix calculation: {total_time/60:.1f} minutes")

def create_sampling_plot(frame_df, sampled_frames, original_frames, output_path, minima_number, system):
    """
    Create a plot showing the sampling strategy used for DCD file creation.
    Shows the total frames and which ones were sampled for inclusion in the DCD file.
    """
    plt.figure(figsize=(12, 8))
    
    # Create a subplot grid with distribution and timeline
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
    
    # RMSD distribution plot
    ax_dist = plt.subplot(gs[0, 1])
    all_rmsd = frame_df['rmsd']
    sampled_rmsd = frame_df[frame_df['frame_number'].isin(sampled_frames)]['rmsd']
    
    ax_dist.hist(all_rmsd, bins=30, orientation='horizontal', alpha=0.5, label='All frames')
    ax_dist.hist(sampled_rmsd, bins=30, orientation='horizontal', alpha=0.7, label='Sampled frames')
    ax_dist.set_ylabel('RMSD (Å)')
    ax_dist.set_xlabel('Count')
    ax_dist.legend(fontsize=8)
    
    # Frame timeline plot
    ax_timeline = plt.subplot(gs[1, 0])
    ax_timeline.scatter(frame_df['frame_number'], frame_df['rmsd'], 
                       c='blue', alpha=0.5, s=2, label='All frames')
    ax_timeline.scatter(sampled_frames, 
                       frame_df[frame_df['frame_number'].isin(sampled_frames)]['rmsd'],
                       c='red', s=10, label='Sampled frames')
    ax_timeline.set_xlabel('Frame number')
    ax_timeline.set_ylabel('RMSD (Å)')
    ax_timeline.legend()
    
    # Frame index heatmap
    ax_index = plt.subplot(gs[0, 0])
    # Create a boolean array showing which frames were selected
    frame_range = np.arange(min(frame_df['frame_number']), max(frame_df['frame_number'])+1)
    sampled_mask = np.zeros(len(frame_range))
    
    # Map the sampled frames to indices in our range
    sample_indices = np.searchsorted(frame_range, sampled_frames)
    # Only use valid indices
    valid_indices = (sample_indices >= 0) & (sample_indices < len(sampled_mask))
    sampled_mask[sample_indices[valid_indices]] = 1
    
    # Reshape for visualization (create a 2D grid)
    grid_width = int(np.sqrt(len(frame_range)) * 1.5)  # Adjust for better aspect ratio
    padding = (grid_width - (len(frame_range) % grid_width)) % grid_width
    grid_data = np.pad(sampled_mask, (0, padding), 'constant')
    grid_height = len(grid_data) // grid_width
    grid_data = grid_data.reshape(grid_height, grid_width)
    
    im = ax_index.imshow(grid_data, cmap='Reds', aspect='auto', interpolation='none')
    ax_index.set_title(f'Frame Selection for Minima {minima_number}\n' +
                      f'Using {len(sampled_frames)} frames from {len(original_frames)} total frames\n' +
                      f'(RMSD range: {all_rmsd.min():.2f} - {all_rmsd.max():.2f} Å)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_index)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Not Selected', 'Selected'])
    
    # Overall stats as text
    stat_text = (
        f"All Frames: {len(original_frames)}\n"
        f"Sampled Frames: {len(sampled_frames)}\n"
        f"Sampling Ratio: {len(sampled_frames)/len(original_frames):.2%}\n\n"
        f"RMSD Statistics:\n"
        f"Original Range: {all_rmsd.min():.3f} - {all_rmsd.max():.3f}\n"
        f"Original Mean±Std: {all_rmsd.mean():.3f}±{all_rmsd.std():.3f}\n"
        f"Sampled Range: {sampled_rmsd.min():.3f} - {sampled_rmsd.max():.3f}\n"
        f"Sampled Mean±Std: {sampled_rmsd.mean():.3f}±{sampled_rmsd.std():.3f}"
    )
    
    plt.figtext(0.75, 0.4, stat_text, fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Sampling strategy plot saved to {output_path}")
    return

def plot_fl_vs_rmsd_with_minima(df, minima_info, output_path, system):
    """Create Free Energy vs RMSD plot with minima marked.
    
    This function:
    1. Removes consecutive points with the same F(λ) raw value to reduce noise
    2. Focuses/zooms on the regions of interest (minima)
    3. Highlights minima with shaded regions and markers
    """
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)
    
    # Remove consecutive points with the same F(λ) raw value (like in process_geo.py)
    df['diff'] = df["F(𝜆) raw"].diff()
    df = df[df['diff'].notna() & (df['diff'] != 0)]
    df = df.drop('diff', axis=1)
    
    # Plot the main data
    ax.plot(df['rmsd'], df['F(𝜆) raw'], 'r-', linewidth=1.5)
    
    # Calculate zoom limits based on minima regions (plus a small margin)
    min_rmsd = min([r[0] for r in minima_info]) - 0.5
    max_rmsd = max([r[1] for r in minima_info]) + 0.5
    
    # Filter to points within the expanded RMSD range for y-axis determination
    zoom_data = df[(df['rmsd'] >= min_rmsd) & (df['rmsd'] <= max_rmsd)]
    if not zoom_data.empty:
        min_energy = zoom_data['F(𝜆) raw'].min() - 1.0
        max_energy = zoom_data['F(𝜆) raw'].max() + 1.0
    else:
        # Fallback if no points in range
        min_energy = df['F(𝜆) raw'].min() - 1.0
        max_energy = df['F(𝜆) raw'].max() + 1.0
    
    # Store minima points for potential additional visualization
    minima_points = []
    
    # Mark minima regions
    for i, min_range in enumerate(minima_info, 1):
        # Highlight the RMSD range region
        well_points = df[
            (df['rmsd'] >= min_range[0]) & 
            (df['rmsd'] <= min_range[1])
        ]
        
        if not well_points.empty:
            rmsd_mid = (min_range[0] + min_range[1]) / 2
            energy_min = well_points['F(𝜆) raw'].min()
            minima_points.append((rmsd_mid, energy_min))
            
            ax.fill_between(well_points['rmsd'], well_points['F(𝜆) raw'], 
                          well_points['F(𝜆) raw'].max(), color='red', alpha=0.2)
            
            ax.scatter(rmsd_mid, energy_min, color='blue', s=100, zorder=5)
            ax.annotate(f'Min {i}', 
                      (rmsd_mid, energy_min),
                      xytext=(10, 10), textcoords='offset points',
                      fontsize=10, color='blue')
    
    # Set axis limits to zoom in
    ax.set_xlim(min_rmsd, max_rmsd)
    ax.set_ylim(min_energy, max_energy)
    
    # Add a small inset showing the full range
    if len(df) > 0:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
        axins.plot(df['rmsd'], df['F(𝜆) raw'], 'r-', linewidth=0.8)
        for rmsd_mid, energy_min in minima_points:
            axins.scatter(rmsd_mid, energy_min, color='blue', s=20)
        axins.set_title("Full Range", fontsize=8)
        
    plt.xlabel('RMSD (Å)', fontsize=12, labelpad=10)
    plt.ylabel('∆G (Kcal/mol)', fontsize=12, labelpad=10)
    plt.title(f'FEL vs RMSD for {system} (with Minima Regions)', fontsize=14)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def create_vmd_script(minima_number, output_dir, frame_files, structure_file, root_path):
    """Create a VMD script to load frames for a specific minimum"""
    # Determine file type
    file_ext = os.path.splitext(structure_file)[1].lower()
    
    # Create path to structure file relative to root_path
    if structure_file.startswith(root_path):
        # Already absolute path
        rel_structure_file = structure_file
    else:
        # Make it absolute
        rel_structure_file = os.path.join(root_path, structure_file)
    
    # Create paths to frame files relative to root_path
    rel_frame_files = []
    for frame_file in frame_files:
        if frame_file.startswith(root_path):
            # Already absolute path
            rel_frame_files.append(frame_file)
        else:
            # Make it absolute
            rel_frame_files.append(os.path.join(root_path, frame_file))
    
    if file_ext == '.psf':
        load_cmd = f"mol new {rel_structure_file} type psf first 0 last -1 step 1 waitfor all"
    else:  # PDB
        load_cmd = f"mol new {rel_structure_file} type pdb first 0 last -1 step 1 waitfor all"
    
    vmd_script = f"""# VMD script for minimum {minima_number}
# Load structure file
{load_cmd}

# Load all frames for this minimum
"""
    for frame_file in rel_frame_files:
        vmd_script += f"mol addfile {frame_file} type dcd first 0 last -1 step 1 waitfor all filebonds 1 autobonds 1 waitfor all\n"

    vmd_script += """
# Set representation
mol modstyle 0 0 NewCartoon 0.300000 10.000000 4.100000 0
mol modcolor 0 0 ColorID 0
mol modmaterial 0 0 Opaque

# Set view
mol showrep 0 0 1
display resetview
display projection orthographic
display depthcue off
display ambientocclusion off
display antialias off
display shadows off

# Set background
color Display Background white

# Set window size
display resize 1024 768

# Set axes
axes location off

# Set labels
label textoffsets 0.0 0.0 0.0
label add Atoms 0 all
label textformat "{%a}"
label textsize 1.0
"""
    
    # Save the VMD script
    vmd_script_path = os.path.join(output_dir, f"minima_{minima_number}.vmd")
    with open(vmd_script_path, 'w') as f:
        f.write(vmd_script)
    print(f"Created VMD script: {vmd_script_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: dcd_analysis.py hosts/hostname/system_name/")
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
        
        # Set the root path
        root_path = "/scratch/users/lm18di/geometry/"

        # Create output directory
        output_dir = os.path.join(root_path, "figures")
        os.makedirs(output_dir, exist_ok=True)

        # Create output directory
        logs_output_dir = os.path.join(root_path, "logs")
        os.makedirs(logs_output_dir, exist_ok=True)
        
        # Set up logging
        logger = setup_logging(logs_output_dir, system)
        logger.info(f"Starting DCD analysis for system: {system}")
        logger.info(f"Output directory: {logs_output_dir}")
        
        # Read the preprocessed data with frame numbers
        dvdl_rmsd_frame_csv = os.path.join(full_path, 'dvdl_rmsd_frame.csv')
        fl_rmsd_csv_path = os.path.join(full_path, 'fl_rmsd.csv')
        
        if not os.path.exists(dvdl_rmsd_frame_csv):
            logger.error("Frame data file not found. Please run find_frames.py first.")
            sys.exit(2)
            
        if not os.path.exists(fl_rmsd_csv_path):
            logger.error("RMSD data files not found. Please run rmsd_values.py first.")
            sys.exit(2)
        
        # Read the data
        dvdl_df = pd.read_csv(dvdl_rmsd_frame_csv)
        fl_df = pd.read_csv(fl_rmsd_csv_path)
        
        # Define minima RMSD ranges determined from Plotly plots from rmsd_values.py
        # Change the ranges for different systems
        minima_ranges = [ 
            # (11.005, 11.35),  # Minimum 1, for bn-bs
            # (15.375, 15.72)   # Minimum 2, for bn-bs
            (4.417, 4.484),  # Minimum 1, for cam-c28w
            (5.422, 5.623),   # Minimum 2, for cam-c28w  
            (6.3265, 6.494),  # Minimum 3, for cam-c28w
            (7.8005, 8.0015),   # Minimum 4, for cam-c28w  
            (10.447, 10.648)  # Minimum 5, for cam-c28w
        ]
        
        logger.info(f"Defined minima ranges: {minima_ranges}")
        
        # Find structure file (PSF or PDB) and DCD files
        structure_file = find_psf_or_pdb_file(full_path)
        if not structure_file:
            logger.error("No PSF or PDB file found")
            sys.exit(2)
        logger.info(f"Using structure file: {structure_file}")
        
        dcd_files = find_all_dcd_files(full_path)
        if not dcd_files:
            logger.error("No valid DCD files found")
            sys.exit(2)
        logger.info(f"Found {len(dcd_files)} DCD files")
        
        # Create Universe
        logger.info("Loading trajectory...")
        u = mda.Universe(structure_file, dcd_files)
        n_atoms = len(u.atoms)
        total_frames = len(u.trajectory)
        logger.info(f"Number of atoms in system: {n_atoms}")
        logger.info(f"Total frames in trajectory: {total_frames}")
        
        # Process each minimum
        for i, rmsd_range in enumerate(minima_ranges, 1):
            logger.info(f"\nProcessing minimum {i} (RMSD range: {rmsd_range[0]:.2f} - {rmsd_range[1]:.2f} Å):")
            
            # Create directory for this minimum's frames
            min_dir = os.path.join(full_path, f"minima_{i}")
            os.makedirs(min_dir, exist_ok=True)
            
            # Find frames where RMSD is within the specified range AND frame number is not NaN
            matching_frames = dvdl_df[
                (dvdl_df['rmsd'] >= rmsd_range[0]) & 
                (dvdl_df['rmsd'] <= rmsd_range[1]) &
                (~pd.isna(dvdl_df['frame_number']))
            ]['frame_number'].unique()
            
            if len(matching_frames) > 0:
                logger.info(f"Found {len(matching_frames)} unique frames")
                frame_numbers = sorted(matching_frames)
                
                # Store original frames before any reduction
                original_frame_numbers = frame_numbers.copy()
                
                # Print initial RMSD statistics for these frames
                frame_data = dvdl_df[dvdl_df['frame_number'].isin(frame_numbers)]
                logger.info("\nInitial RMSD Statistics:")
                logger.info(f"RMSD range: {frame_data['rmsd'].min():.3f} to {frame_data['rmsd'].max():.3f}")
                logger.info(f"RMSD mean ± std: {frame_data['rmsd'].mean():.3f} ± {frame_data['rmsd'].std():.3f}")
                
                # Verify all frames are within RMSD range
                out_of_range = frame_data[
                    (frame_data['rmsd'] < rmsd_range[0]) | 
                    (frame_data['rmsd'] > rmsd_range[1])
                ]
                if not out_of_range.empty:
                    logger.warning(f"Found {len(out_of_range)} frames outside RMSD range!")
                    logger.warning("These frames will be excluded.")
                    frame_numbers = frame_data[
                        (frame_data['rmsd'] >= rmsd_range[0]) & 
                        (frame_data['rmsd'] <= rmsd_range[1])
                    ]['frame_number'].unique()
                    frame_data = frame_data[frame_data['frame_number'].isin(frame_numbers)]
                    logger.info(f"Remaining frames: {len(frame_numbers)}")
                
                # If we have more than 2500 frames, use RMSD-aware sampling
                MAX_FRAMES = 2500
                if len(frame_numbers) > MAX_FRAMES:
                    logger.info(f"Number of frames ({len(frame_numbers)}) exceeds limit of {MAX_FRAMES}.")
                    logger.info("Using RMSD-aware sampling to select representative frames...")
                    
                    temp_df = frame_data.copy()
                    frame_numbers = sample_frames_smart(temp_df, MAX_FRAMES, rmsd_col='rmsd')
                    
                    # Print sampling statistics
                    original_rmsd = temp_df['rmsd']
                    sampled_frames = temp_df[temp_df['frame_number'].isin(frame_numbers)]
                    sampled_rmsd = sampled_frames['rmsd']
                    
                    logger.info("\nSampling Statistics:")
                    logger.info(f"Original frame count: {len(temp_df['frame_number'].unique())}")
                    logger.info(f"Sampled frame count: {len(frame_numbers)}")
                    logger.info(f"Original RMSD range: {original_rmsd.min():.3f} to {original_rmsd.max():.3f}")
                    logger.info(f"Sampled RMSD range: {sampled_rmsd.min():.3f} to {sampled_rmsd.max():.3f}")
                    logger.info(f"Original RMSD mean ± std: {original_rmsd.mean():.3f} ± {original_rmsd.std():.3f}")
                    logger.info(f"Sampled RMSD mean ± std: {sampled_rmsd.mean():.3f} ± {sampled_rmsd.std():.3f}")
                    
                else:
                    logger.info(f"Number of frames ({len(frame_numbers)}) is within limit. Using all frames.")
                
                # Always create the sampling plot
                sampling_plot = os.path.join(output_dir, f"{system}_minima_{i}_sampling_for_dcd.pdf")
                create_sampling_plot(frame_data, frame_numbers, original_frame_numbers, sampling_plot, i, system)
                
                # Ensure frame numbers are within trajectory bounds
                valid_frames = [int(f) for f in frame_numbers if 0 <= f < total_frames]
                
                if len(valid_frames) < len(frame_numbers):
                    logger.warning(f"{len(frame_numbers) - len(valid_frames)} frames were outside trajectory bounds")
                    logger.warning(f"Original frame range: {min(frame_numbers)} to {max(frame_numbers)}")
                    logger.warning(f"Valid frame range: {min(valid_frames)} to {max(valid_frames)}")
                    logger.warning(f"Total frames in trajectory: {total_frames}")
                
                if valid_frames:
                    logger.info(f"\nFrame distribution analysis:")
                    logger.info(f"Frame range: {min(valid_frames)} to {max(valid_frames)}")
                    logger.info(f"Total frames to extract: {len(valid_frames)}")
                    
                    # Analyze gaps between consecutive frames
                    frame_gaps = np.diff(valid_frames)
                    consecutive_frames = np.sum(frame_gaps == 1)
                    logger.info(f"\nFrame gap analysis:")
                    logger.info(f"Number of consecutive frame pairs: {consecutive_frames}")
                    logger.info(f"Average gap between frames: {np.mean(frame_gaps):.2f}")
                    logger.info(f"Median gap between frames: {np.median(frame_gaps):.2f}")
                    logger.info(f"Largest gap between frames: {np.max(frame_gaps)}")
                    logger.info(f"Number of gaps > 100 frames: {np.sum(frame_gaps > 100)}")
                    
                    # Extract frames to a single DCD file
                    output_file = os.path.join(min_dir, f"minimum_{i}_frames.dcd")
                    write_frames_to_dcd(u, valid_frames, output_file)
                    
                    # Create VMD script for this minimum
                    create_vmd_script(i, min_dir, [output_file], structure_file, root_path)
                    
                    # Create RMSD matrix plot
                    rmsd_matrix_plot = os.path.join(output_dir, f"{system}_minima_{i}_2d_rmsd_plot.pdf")
                    create_rmsd_matrix(u, original_frame_numbers, rmsd_matrix_plot, i, dvdl_df, logger)
                else:
                    logger.error("No valid frames found within trajectory bounds")
        
        # Create final summary plots
        # rmsd_plot = os.path.join(output_dir, f"{system}_rmsd_with_minima.pdf")
        fl_plot = os.path.join(output_dir, f"{system}_fl_with_minima.pdf")
        
        # plot_rmsd_vs_time_with_minima(dvdl_df, minima_ranges, rmsd_plot, system)
        plot_fl_vs_rmsd_with_minima(fl_df, minima_ranges, fl_plot, system)
        
        logger.info("Analysis completed successfully")

if __name__ == '__main__':
    main()
