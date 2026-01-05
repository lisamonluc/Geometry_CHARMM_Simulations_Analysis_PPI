#!/usr/bin/python3

# usage: find_minima.py hosts/hostname/system_name/

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import MDAnalysis as mda
from scipy.signal import find_peaks

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
    """Find all valid DCD files in the base directory and its subdirectories"""
    print("Searching for DCD files (this may take a while for large directory trees)...")
    sys.stdout.flush()
    
    dcd_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.dcd'):
                dcd_path = os.path.join(root, file)
                if is_empty_dcd(dcd_path):
                    print(f"Skipping empty/corrupt DCD file: {dcd_path}")
                    continue
                dcd_files.append(dcd_path)
                print(f"Found valid DCD file: {dcd_path}")
                sys.stdout.flush()
    
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

def find_energy_minima(df, prominence=0.5):
    """Find energy minima in the free energy landscape"""
    inverted_energy = -df['F(𝜆) raw']
    peaks, properties = find_peaks(inverted_energy, prominence=prominence)
    
    minima = []
    for peak in peaks:
        rmsd = df['rmsd'].iloc[peak]
        energy = df['F(𝜆) raw'].iloc[peak]
        minima.append((rmsd, energy))
    
    return minima

def get_frame_from_time(time_fs, freq=4000):
    """Convert time in femtoseconds to frame number"""
    return int(np.round(time_fs / freq))

def get_closest_frames(time_fs, freq=4000, window=10):
    """Get frame numbers closest to the 4000-timestep intervals within a window of ±10 frames"""
    base_frame = int(np.round(time_fs / freq))
    return list(range(max(0, base_frame - window), base_frame + window + 1))

def analyze_energy_well(df, rmsd_min, energy_min):
    """Analyze the characteristics of an energy well around a minimum"""
    # Sort the data by RMSD for easier analysis
    sorted_df = df.sort_values('rmsd')
    
    # Find the index of the minimum
    min_idx = sorted_df[sorted_df['rmsd'] == rmsd_min].index[0]
    
    # Calculate the energy gradient
    sorted_df['energy_gradient'] = sorted_df['F(𝜆) raw'].diff()
    
    # Look for basin boundaries by finding significant changes in gradient
    left_boundary = None
    right_boundary = None
    
    # Use different thresholds based on the minimum's RMSD
    if rmsd_min > 14:  # Main minima around 15.72
        gradient_threshold = 0.2  # Stricter threshold for main minima
        max_width = 2.0  # Maximum basin width in Å
        energy_threshold = 1.0  # Maximum barrier height in kcal/mol
    else:  # Small minima around 12
        gradient_threshold = 0.1  # Even stricter threshold for small minima
        max_width = 1.0  # Maximum basin width in Å
        energy_threshold = 0.5  # Maximum barrier height in kcal/mol
    
    # Look to the left of the minimum
    for i in range(min_idx - 1, -1, -1):
        if abs(sorted_df['energy_gradient'].iloc[i]) > gradient_threshold or \
           abs(sorted_df['F(𝜆) raw'].iloc[i] - energy_min) > energy_threshold or \
           abs(sorted_df['rmsd'].iloc[i] - rmsd_min) > max_width:
            left_boundary = sorted_df['rmsd'].iloc[i]
            break
    
    # Look to the right of the minimum
    for i in range(min_idx + 1, len(sorted_df)):
        if abs(sorted_df['energy_gradient'].iloc[i]) > gradient_threshold or \
           abs(sorted_df['F(𝜆) raw'].iloc[i] - energy_min) > energy_threshold or \
           abs(sorted_df['rmsd'].iloc[i] - rmsd_min) > max_width:
            right_boundary = sorted_df['rmsd'].iloc[i]
            break
    
    # If we couldn't find clear boundaries, use a fallback method
    if left_boundary is None or right_boundary is None:
        fallback_range = 0.1  # ±0.1 Å around the minimum
        
        if left_boundary is None:
            left_boundary = max(rmsd_min - fallback_range, sorted_df['rmsd'].min())
        if right_boundary is None:
            right_boundary = min(rmsd_min + fallback_range, sorted_df['rmsd'].max())
    
    # Calculate the basin width
    basin_width = right_boundary - left_boundary
    
    # Calculate 20% of the basin width
    basin_20_percent = basin_width * 0.2
    
    # Find points within 20% of the basin width around the minimum
    rmsd_min_range = rmsd_min - (basin_20_percent / 2)
    rmsd_max_range = rmsd_min + (basin_20_percent / 2)
    
    # Get points within this RMSD range
    nearby_points = df[
        (df['rmsd'] >= rmsd_min_range) & 
        (df['rmsd'] <= rmsd_max_range)
    ]
    
    # Calculate energy range within the basin
    basin_points = df[
        (df['rmsd'] >= left_boundary) & 
        (df['rmsd'] <= right_boundary)
    ]
    basin_energy_range = basin_points['F(𝜆) raw'].max() - basin_points['F(𝜆) raw'].min()
    
    return {
        'well_depth': basin_energy_range,
        'well_width': basin_width,
        'min_energy': energy_min,
        'basin_20_percent_width': basin_20_percent,
        'points_in_well': len(nearby_points),
        'rmsd_range': (rmsd_min_range, rmsd_max_range),
        'basin_boundaries': (left_boundary, right_boundary),
        'gradient_threshold': gradient_threshold,
        'used_fallback': left_boundary is None or right_boundary is None
    }

def plot_energy_well(df, rmsd_min, energy_min, well_info, output_path, frame_numbers=None):
    """Create a detailed plot of the energy well around a minimum"""
    plt.figure(figsize=(12, 8))
    
    # Create subplots for energy and gradient
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot the energy landscape
    ax1.plot(df['rmsd'], df['F(𝜆) raw'], 'b-', label='Energy Landscape')
    
    # Highlight the basin region
    basin_points = df[
        (df['rmsd'] >= well_info['basin_boundaries'][0]) & 
        (df['rmsd'] <= well_info['basin_boundaries'][1])
    ]
    ax1.fill_between(basin_points['rmsd'], basin_points['F(𝜆) raw'], 
                    basin_points['F(𝜆) raw'].max(), color='lightblue', alpha=0.3)
    
    # Highlight the 20% region
    well_points = df[
        (df['rmsd'] >= well_info['rmsd_range'][0]) & 
        (df['rmsd'] <= well_info['rmsd_range'][1])
    ]
    ax1.fill_between(well_points['rmsd'], well_points['F(𝜆) raw'], 
                    well_points['F(𝜆) raw'].max(), color='red', alpha=0.2)
    
    # Mark the minimum
    ax1.scatter(rmsd_min, energy_min, color='red', s=100, label='Minimum')
    
    # Add basin boundaries
    if well_info['basin_boundaries'][0]:
        ax1.axvline(x=well_info['basin_boundaries'][0], color='gray', linestyle='--', 
                   label='Basin Boundary')
    if well_info['basin_boundaries'][1]:
        ax1.axvline(x=well_info['basin_boundaries'][1], color='gray', linestyle='--')
    
    # Plot the energy gradient
    sorted_df = df.sort_values('rmsd')
    sorted_df['energy_gradient'] = sorted_df['F(𝜆) raw'].diff()
    ax2.plot(sorted_df['rmsd'], sorted_df['energy_gradient'], 'g-', label='Energy Gradient')
    ax2.axhline(y=well_info['gradient_threshold'], color='gray', linestyle=':', 
               label='Gradient Threshold')
    ax2.axhline(y=-well_info['gradient_threshold'], color='gray', linestyle=':')
    
    # Add annotations
    annotation_text = f'Basin Width: {well_info["well_width"]:.2f} Å\n'
    annotation_text += f'20% Basin Width: {well_info["basin_20_percent_width"]:.2f} Å\n'
    annotation_text += f'Well Depth: {well_info["well_depth"]:.2f} kcal/mol'
    if well_info['used_fallback']:
        annotation_text += '\n(Using fallback range)'
    
    ax1.annotate(annotation_text,
                xy=(rmsd_min, energy_min),
                xytext=(10, 10), textcoords='offset points')
    
    # Add frame numbers if provided
    if frame_numbers:
        frame_points = df[df.index.isin(frame_numbers)]
        for idx, row in frame_points.iterrows():
            ax1.annotate(f'Frame {idx}',
                        xy=(row['rmsd'], row['F(𝜆) raw']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='black')
    
    ax1.set_ylabel('Free Energy (kcal/mol)')
    ax2.set_ylabel('Energy Gradient')
    ax2.set_xlabel('RMSD (Å)')
    ax1.set_title('Energy Well Analysis')
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_rmsd_matrix(u, frame_numbers, output_path, minima_number):
    """Create a pairwise RMSD matrix for given frames and save as a plot"""
    n_frames = len(frame_numbers)
    rmsd_matrix = np.zeros((n_frames, n_frames))
    
    # Calculate RMSD between each pair of frames
    print(f"Calculating RMSD matrix for {n_frames} frames...")
    for i, frame1 in enumerate(frame_numbers):
        u.trajectory[frame1]
        for j, frame2 in enumerate(frame_numbers):
            if i <= j:  # Only calculate upper triangle
                u.trajectory[frame2]
                rmsd = mda.analysis.rms.rmsd(u.atoms.positions, u.atoms.positions, center=True, superposition=True)
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd  # Symmetric matrix
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(rmsd_matrix, cmap='viridis', aspect='equal')
    plt.colorbar(label='RMSD (Å)')
    
    # Add frame numbers to axes
    plt.xticks(range(n_frames), [f'Frame {f}' for f in frame_numbers], rotation=45)
    plt.yticks(range(n_frames), [f'Frame {f}' for f in frame_numbers])
    
    plt.title(f'Pairwise RMSD Matrix - Minimum {minima_number}', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"RMSD matrix plot saved to {output_path}")

def plot_rmsd_vs_time_with_minima(df, minima_info, output_path, system):
    """Create RMSD vs Time plot with minima marked"""
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)
    
    # Plot the main data
    ax.plot(df['t_fs'], df['rmsd'], 'b-', linewidth=1.5)
    
    # Mark minima
    for min_info in minima_info:
        i = min_info['index']
        rmsd = min_info['rmsd']
        well_info = min_info['well_info']
        
        # Use the 20% basin width range to find matching times
        matching_times = df[
            (df['rmsd'] >= well_info['rmsd_range'][0]) & 
            (df['rmsd'] <= well_info['rmsd_range'][1])
        ]['t_fs']
        
        if not matching_times.empty:
            # Plot points for each minimum
            ax.scatter(matching_times, [rmsd] * len(matching_times), 
                     color='red', s=100, zorder=5)
            # Add labels
            for t in matching_times:
                ax.annotate(f'Min {i+1}', 
                          (t, rmsd),
                          xytext=(10, 10), textcoords='offset points',
                          fontsize=10, color='red')
    
    plt.xlabel('Time (fs)', fontsize=12, labelpad=10)
    plt.ylabel('RMSD (Å)', fontsize=12, labelpad=10)
    plt.title(f'RMSD vs Time for {system} (with Energy Minima)', fontsize=14)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_fl_vs_rmsd_with_minima(df, minima_info, output_path, system):
    """Create Free Energy vs RMSD plot with minima marked"""
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)
    
    # Plot the main data
    ax.plot(df['rmsd'], df['F(𝜆) raw'], 'r-', linewidth=1.5)
    
    # Mark minima
    for min_info in minima_info:
        i = min_info['index']
        rmsd = min_info['rmsd']
        energy = min_info['energy']
        well_info = min_info['well_info']
        
        # Plot the minimum point
        ax.scatter(rmsd, energy, color='blue', s=100, zorder=5)
        ax.annotate(f'Min {i+1}', 
                  (rmsd, energy),
                  xytext=(10, 10), textcoords='offset points',
                  fontsize=10, color='blue')
        
        # Highlight the basin region
        basin_points = df[
            (df['rmsd'] >= well_info['basin_boundaries'][0]) & 
            (df['rmsd'] <= well_info['basin_boundaries'][1])
        ]
        ax.fill_between(basin_points['rmsd'], basin_points['F(𝜆) raw'], 
                       basin_points['F(𝜆) raw'].max(), color='lightblue', alpha=0.3)
        
        # Highlight the 20% region
        well_points = df[
            (df['rmsd'] >= well_info['rmsd_range'][0]) & 
            (df['rmsd'] <= well_info['rmsd_range'][1])
        ]
        ax.fill_between(well_points['rmsd'], well_points['F(𝜆) raw'], 
                       well_points['F(𝜆) raw'].max(), color='red', alpha=0.2)
    
    plt.xlabel('RMSD (Å)', fontsize=12, labelpad=10)
    plt.ylabel('∆G (Kcal/mol)', fontsize=12, labelpad=10)
    plt.title(f'FEL vs RMSD for {system} (with Energy Minima)', fontsize=14)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def create_vmd_script(minima_number, output_dir, system, frame_files, structure_file):
    """Create a VMD script to load frames for a specific minimum"""
    # Determine file type
    file_ext = os.path.splitext(structure_file)[1].lower()
    if file_ext == '.psf':
        load_cmd = f"mol new {structure_file} type psf first 0 last -1 step 1 waitfor all"
    else:  # PDB
        load_cmd = f"mol new {structure_file} type pdb first 0 last -1 step 1 waitfor all"
    
    vmd_script = f"""# VMD script for minimum {minima_number}
# Load structure file
{load_cmd}

# Load all frames for this minimum
"""
    for frame_file in frame_files:
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
        print("Usage: find_minima.py hosts/hostname/system_name/")
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
        
        # Read the RMSD data
        fl_rmsd_csv_path = os.path.join(full_path, 'fl_rmsd.csv')
        dvdl_rmsd_csv_path = os.path.join(full_path, 'dvdl_rmsd.csv')
        
        if not os.path.exists(fl_rmsd_csv_path) or not os.path.exists(dvdl_rmsd_csv_path):
            print("Error: RMSD data files not found. Please run rmsd_values.py first.")
            sys.exit(2)
        
        # Read the data
        fl_df = pd.read_csv(fl_rmsd_csv_path)
        dvdl_df = pd.read_csv(dvdl_rmsd_csv_path)
        
        # Find energy minima
        minima = find_energy_minima(fl_df)
        print("\nFound energy minima:")
        for i, (rmsd, energy) in enumerate(minima):
            print(f"Minimum {i+1}: RMSD = {rmsd:.2f} Å, Energy = {energy:.2f} kcal/mol")
        
        # Create output directory
        output_dir = "/scratch/users/lm18di/geometry/figures/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze each minimum and store results
        minima_info = []
        for i, (rmsd, energy) in enumerate(minima):
            well_info = analyze_energy_well(fl_df, rmsd, energy)
            if well_info:
                minima_info.append({
                    'index': i,
                    'rmsd': rmsd,
                    'energy': energy,
                    'well_info': well_info
                })
        
        # Find structure file (PSF or PDB) and DCD files
        structure_file = find_psf_or_pdb_file(full_path)
        if not structure_file:
            print("Error: No PSF or PDB file found")
            sys.exit(2)
        
        dcd_files = find_all_dcd_files(full_path)
        if not dcd_files:
            print("Error: No valid DCD files found")
            sys.exit(2)
        
        # Create Universe
        print("\nLoading trajectory...")
        u = mda.Universe(structure_file, dcd_files)
        
        # Get number of atoms
        n_atoms = len(u.atoms)
        print(f"Number of atoms in system: {n_atoms}")
        
        # Process each minimum
        for min_info in minima_info:
            i = min_info['index']
            rmsd = min_info['rmsd']
            energy = min_info['energy']
            well_info = min_info['well_info']
            
            print(f"\nProcessing minimum {i+1} (RMSD = {rmsd:.2f} Å, Energy = {energy:.2f} kcal/mol):")
            print(f"Well Depth: {well_info['well_depth']:.2f} kcal/mol")
            print(f"Well Width: {well_info['well_width']:.2f} Å")
            print(f"Points in Well: {well_info['points_in_well']}")
            
            # Create directory for this minimum's frames
            min_dir = os.path.join(full_path, f"minima_{i+1}")
            os.makedirs(min_dir, exist_ok=True)
            
            # Find times where RMSD is within the well's RMSD range
            matching_times = dvdl_df[
                (dvdl_df['rmsd'] >= well_info['rmsd_range'][0]) & 
                (dvdl_df['rmsd'] <= well_info['rmsd_range'][1])
            ]['t_fs']
            
            if not matching_times.empty:
                # Get frames close to 4000-timestep intervals
                frame_sets = [get_closest_frames(t) for t in matching_times]
                frame_numbers = sorted(list(set([f for frames in frame_sets for f in frames])))
                
                print(f"Extracting frames {min(frame_numbers)} to {max(frame_numbers)}")
                
                # Extract frames
                frame_files = []
                for frame in frame_numbers:
                    output_file = os.path.join(min_dir, f"frame_{frame}.dcd")
                    with mda.Writer(output_file, n_atoms=n_atoms) as w:
                        u.trajectory[frame]
                        w.write(u)
                    frame_files.append(output_file)
                
                # Create VMD script for this minimum
                create_vmd_script(i+1, min_dir, system, frame_files, structure_file)
                
                # Create RMSD matrix plot
                rmsd_matrix_plot = os.path.join(output_dir, f"{system}_minima_{i+1}_2d_rmsd_plot.png")
                create_rmsd_matrix(u, frame_numbers, rmsd_matrix_plot, i+1)
                
                # Create energy well plot with frame numbers
                well_plot = os.path.join(output_dir, f"{system}_minima_{i+1}_energy_well.png")
                plot_energy_well(fl_df, rmsd, energy, well_info, well_plot, frame_numbers)
        
        # Create final summary plots
        rmsd_plot = os.path.join(output_dir, f"{system}_rmsd_with_minima.pdf")
        fl_plot = os.path.join(output_dir, f"{system}_fl_with_minima.pdf")
        
        plot_rmsd_vs_time_with_minima(dvdl_df, minima_info, rmsd_plot, system)
        plot_fl_vs_rmsd_with_minima(fl_df, minima_info, fl_plot, system)

if __name__ == '__main__':
    main() 