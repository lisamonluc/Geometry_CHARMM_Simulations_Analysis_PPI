import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import itertools
import warnings
import logging
from tqdm.auto import tqdm

# --- Constants ---
PDB_FILE = '/scratch/users/lm18di/geometry/hosts/nerve/bn_bs/eq.pdb'
DCD_BASE_DIR = '/scratch/users/lm18di/geometry/hosts/nerve/bn_bs/'
CSV_OUTPUT_DIR = '/scratch/users/lm18di/geometry/interactions_csv/'
PLOT_OUTPUT_DIR = '/scratch/users/lm18di/geometry/interactions_plots/'
NO_CONTACT_FILE = os.path.join(PLOT_OUTPUT_DIR, 'no_contact_frames.txt')

# Interaction Criteria
HBOND_DIST_CUTOFF = 3.5
SALT_BRIDGE_DIST_CUTOFF = 4.0
VDW_DIST_CUTOFF = 3.8
FRAME_STEP = 10 # Analyze every Nth frame

# Atom Selections
PROC_SEL = "segid PROC"
PROF_SEL = "segid PROF"

# Salt Bridge Selections (using atom names as provided)
POS_CHARGE_SEL = "(resname ARG and name NH1 NH2 NE) or (resname LYS and name NZ) or (resname HSP and name NE2)"
NEG_CHARGE_SEL = "(resname GLU and name OE1 OE2) or (resname ASP and name OD1 OD2)"

# VDW Selections (Carbon atoms in specified hydrophobic residues)
HYDROPHOBIC_RES = "GLY ALA VAL LEU ILE MET PRO PHE TRP SER" # Including SER as requested
VDW_ATOM_SEL = f"name C* and resname {' '.join(HYDROPHOBIC_RES.split())}"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Helper Functions for DCD Loading (from dcd_analysis.py) ---
def is_empty_dcd(dcd_path):
    """Check if a DCD file is empty or corrupted"""
    try:
        file_size = os.path.getsize(dcd_path)
        if file_size < 100: # Basic check for minimal DCD header size
            return True

        # Use MDAnalysis with suppress_warnings=True to avoid printing warnings for every file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Check if universe can be created and has frames without full iteration
            try:
                u = mda.Universe(dcd_path)
                if len(u.trajectory) == 0:
                    return True
                # Try accessing the first frame's existence, not its data
                if not hasattr(u.trajectory, '__len__') or len(u.trajectory) < 1:
                     return True # Should not happen if len > 0, but belt and suspenders
            except Exception:
                 # If universe creation or basic checks fail, consider it problematic
                 return True
        return False # If all checks pass
    except OSError: # File not found or inaccessible
        return True
    except Exception as e:
        # Catch other potential issues during size check or initial access
        logger.warning(f"Error during initial check of DCD file {dcd_path}: {str(e)}")
        return True

def get_restart_folder_depth(path):
    """Calculate how many restart folders deep this path is"""
    parts = os.path.normpath(path).split(os.sep)
    # Count occurrences of 'restart' that are full directory names
    restart_count = sum(1 for part in parts if part == 'restart')
    return restart_count

def find_all_dcd_files(base_dir):
    """Find all valid DCD files in the base directory and its subdirectories, excluding minima_* directories"""
    logger.info(f"Searching for DCD files in {base_dir} (excluding minima_*/)...")

    dcd_files = []
    processed_count = 0
    skipped_empty_count = 0
    skipped_minima_count = 0

    for root, dirs, files in os.walk(base_dir, topdown=True):
        # Filter out minima_* directories from further traversal
        dirs[:] = [d for d in dirs if not d.startswith('minima_')]

        # Check if the current root itself is inside a minima_* directory (needed if walk started above minima)
        if 'minima_' in root.split(os.sep):
             continue # Skip processing files in this directory

        for file in files:
            processed_count += 1
            if file.endswith('.dcd'):
                dcd_path = os.path.join(root, file)

                # Explicitly check path components again, robustness for complex structures
                if any(part.startswith('minima_') for part in dcd_path.split(os.sep)):
                    # logger.info(f"Skipping due to minima path: {dcd_path}") # Can be verbose
                    skipped_minima_count += 1
                    continue

                if is_empty_dcd(dcd_path):
                    # logger.info(f"Skipping empty/corrupt DCD file: {dcd_path}") # Can be verbose
                    skipped_empty_count += 1
                    continue

                dcd_files.append(dcd_path)

            # Log progress less frequently
            if processed_count > 0 and processed_count % 20000 == 0:
                 logger.info(f"  Scanned {processed_count} potential files...")


    logger.info(f"\nFinished DCD search:")
    logger.info(f"- Processed {processed_count} total entries.")
    logger.info(f"- Found {len(dcd_files)} valid DCD files.")
    logger.info(f"- Skipped {skipped_empty_count} empty/corrupt files.")
    logger.info(f"- Skipped {skipped_minima_count} files in minima_* directories.")

    if not dcd_files:
         logger.warning("No valid DCD files found.")
         return []

    logger.info("Sorting DCD files...")
    dcd_files.sort(key=lambda path: (get_restart_folder_depth(path), path))
    logger.info("DCD files sorted.")

    return dcd_files


# --- Interaction Analysis Functions ---

def find_hydrogen_bonds(universe, proc_sel, prof_sel, dist_cutoff):
    """Find hydrogen bonds between PROC and PROF using MDAnalysis HydrogenBondAnalysis."""
    logger.info("\nRunning Hydrogen Bond analysis (PROC <-> PROF only)...")
    # Define potential donors and acceptors *within* each segment
    proc_donors_sel = f"({proc_sel}) and (name N* O*)"
    proc_hydrogens_sel = f"({proc_sel}) and name H* and bonded ({proc_donors_sel})"
    proc_acceptors_sel = f"({proc_sel}) and (name N* O*)"

    prof_donors_sel = f"({prof_sel}) and (name N* O*)"
    prof_hydrogens_sel = f"({prof_sel}) and name H* and bonded ({prof_donors_sel})"
    prof_acceptors_sel = f"({prof_sel}) and (name N* O*)"

    # Run H-bond analysis twice: PROC->PROF and PROF->PROC
    hbonds_proc_prof = []
    hbonds_prof_proc = []

    try:
        logger.info("Running PROC donors -> PROF acceptors...")
        hbond_analysis_p2f = HydrogenBondAnalysis(
            universe=universe,
            donors_sel=proc_donors_sel,
            hydrogens_sel=proc_hydrogens_sel,
            acceptors_sel=prof_acceptors_sel, # PROC donors, PROF acceptors
            d_a_cutoff=dist_cutoff,
            d_h_a_angle_cutoff=120.0, # Common angle cutoff
            update_selections=False # Selections are static
        )
        # MDA's run method has its own verbose option which might print progress
        hbond_analysis_p2f.run(step=FRAME_STEP, verbose=True)
        hbonds_proc_prof = hbond_analysis_p2f.results.hbonds
        logger.info(f"Found {len(hbonds_proc_prof)} PROC->PROF H-bond instances.")

        logger.info("Running PROF donors -> PROC acceptors...")
        hbond_analysis_f2p = HydrogenBondAnalysis(
            universe=universe,
            donors_sel=prof_donors_sel,
            hydrogens_sel=prof_hydrogens_sel,
            acceptors_sel=proc_acceptors_sel, # PROF donors, PROC acceptors
            d_a_cutoff=dist_cutoff,
            d_h_a_angle_cutoff=120.0,
            update_selections=False
        )
        hbond_analysis_f2p.run(step=FRAME_STEP, verbose=True)
        hbonds_prof_proc = hbond_analysis_f2p.results.hbonds
        logger.info(f"Found {len(hbonds_prof_proc)} PROF->PROC H-bond instances.")

    except Exception as e:
         logger.error(f"Error during HydrogenBondAnalysis setup or run: {e}", exc_info=True)
         return defaultdict(list) # Return empty dict on error

    # Process results: Filter for PROC <-> PROF interactions and aggregate minimum distance per pair per frame
    # Results columns: frame, donor_idx, hydrogen_idx, acceptor_idx, distance (D..A), angle (D-H..A)
    # No filtering needed now as analysis was specific, just combine and process

    hb_data = defaultdict(lambda: defaultdict(lambda: float('inf'))) # {(proc_res, prof_res): {frame: min_dist}}

    # Combine results from both runs
    all_interprotein_hbonds = np.concatenate((hbonds_proc_prof, hbonds_prof_proc)) if len(hbonds_proc_prof) > 0 and len(hbonds_prof_proc) > 0 else (hbonds_proc_prof if len(hbonds_proc_prof) > 0 else hbonds_prof_proc)
    n_hbonds_found = len(all_interprotein_hbonds)
    logger.info(f"Processing {n_hbonds_found} total inter-protein H-bond instances...")

    # Wrap hbond results iteration with tqdm
    processed_hbonds = 0
    skipped_invalid_idx = 0
    for row in tqdm(all_interprotein_hbonds, total=n_hbonds_found, desc="Processing H-bonds", unit="hbonds"):
        # Unpack row data
        frame, d_idx_raw, _, a_idx_raw, dist, _ = row # Unpack raw indices

        # Check if indices are numeric (int or float) and cast to int
        d_idx = None
        a_idx = None
        valid_indices = True
        try:
            if isinstance(d_idx_raw, (int, np.integer, float, np.floating)):
                d_idx = int(d_idx_raw)
            else:
                valid_indices = False

            if isinstance(a_idx_raw, (int, np.integer, float, np.floating)):
                a_idx = int(a_idx_raw)
            else:
                valid_indices = False

            # Check if cast resulted in valid indices (e.g. NaN float becomes bad int)
            # This check might be redundant if the isinstance covers it, but safer.
            if d_idx is None or a_idx is None:
                 valid_indices = False

        except (ValueError, TypeError):
             valid_indices = False # Catch potential errors during casting

        if not valid_indices:
            # Limit logging frequency for this warning if it's common
            if skipped_invalid_idx < 10:
                logger.warning(f"Skipping H-bond record at frame {frame} due to invalid index type/value: d_idx_raw={d_idx_raw}({type(d_idx_raw)}), a_idx_raw={a_idx_raw}({type(a_idx_raw)}) ")
            elif skipped_invalid_idx == 10:
                logger.warning("Further invalid index warnings will be suppressed...")
            skipped_invalid_idx += 1
            continue

        # --- Indices are valid integers, proceed ---

        # Determine PROC and PROF residues based on indices
        # Note: We don't strictly need the proc_indices/prof_indices sets anymore
        # because the analysis runs were specific, but we need to know which is which.
        # It's safer to look up the segid.
        try:
             donor_atom = universe.atoms[d_idx]
             acceptor_atom = universe.atoms[a_idx]

             # Check if donor is PROC and acceptor is PROF, or vice versa
             if donor_atom.segid == 'PROC' and acceptor_atom.segid == 'PROF':
                  proc_res = f"{donor_atom.resname.lower()}{donor_atom.resid}"
                  prof_res = f"{acceptor_atom.resname.lower()}{acceptor_atom.resid}"
             elif donor_atom.segid == 'PROF' and acceptor_atom.segid == 'PROC':
                  proc_res = f"{acceptor_atom.resname.lower()}{acceptor_atom.resid}" # PROC is acceptor
                  prof_res = f"{donor_atom.resname.lower()}{donor_atom.resid}" # PROF is donor
             else:
                  # This shouldn't happen given the analysis setup, but catch it.
                  if skipped_invalid_idx < 10: # Reuse counter for logging limit
                       logger.warning(f"Skipping H-bond with unexpected segids: D:{donor_atom.segid} A:{acceptor_atom.segid}")
                  elif skipped_invalid_idx == 10:
                       logger.warning("Further unexpected segid warnings suppressed...")
                  skipped_invalid_idx += 1
                  continue

        except IndexError: # Catch if index is somehow still out of bounds
             if skipped_invalid_idx < 10:
                  logger.warning(f"Skipping H-bond record at frame {frame} due to out-of-bounds index: d_idx={d_idx}, a_idx={a_idx}")
             elif skipped_invalid_idx == 10:
                  logger.warning("Further out-of-bounds index warnings suppressed...")
             skipped_invalid_idx += 1
             continue

        pair = (proc_res, prof_res)
        # Update the minimum distance found for this pair in this frame
        if dist < hb_data[pair][frame]:
            hb_data[pair][frame] = dist
        processed_hbonds += 1 # Count successfully processed bonds

    # Convert to the final format: {pair: [(frame, min_dist)] sorted by frame}
    logger.info(f"Finished processing H-bonds. Successfully processed: {processed_hbonds}. Skipped due to invalid indices: {skipped_invalid_idx}.")
    final_hb_data = defaultdict(list)
    for pair, frame_data in hb_data.items():
        final_hb_data[pair] = sorted(frame_data.items())

    logger.info(f"Processed H-bond data for {len(final_hb_data)} unique PROC-PROF residue pairs.")
    return final_hb_data


def find_salt_bridges(universe, proc_sel, prof_sel, pos_charge_sel, neg_charge_sel, dist_cutoff):
    logger.info("\nFinding Salt Bridges...")
    # Define the four atom groups based on segment and charge selection
    proc_pos_atoms = universe.select_atoms(f"({proc_sel}) and ({pos_charge_sel})")
    proc_neg_atoms = universe.select_atoms(f"({proc_sel}) and ({neg_charge_sel})")
    prof_pos_atoms = universe.select_atoms(f"({prof_sel}) and ({pos_charge_sel})")
    prof_neg_atoms = universe.select_atoms(f"({prof_sel}) and ({neg_charge_sel})")

    logger.info(f"Selected atoms - PROC(+): {len(proc_pos_atoms)}, PROC(-): {len(proc_neg_atoms)}, PROF(+): {len(prof_pos_atoms)}, PROF(-): {len(prof_neg_atoms)}")

    # Check if we have atoms in potentially interacting groups
    has_proc_pos_prof_neg = len(proc_pos_atoms) > 0 and len(prof_neg_atoms) > 0
    has_proc_neg_prof_pos = len(proc_neg_atoms) > 0 and len(prof_pos_atoms) > 0

    if not (has_proc_pos_prof_neg or has_proc_neg_prof_pos):
         logger.warning("No potentially interacting charged groups found. Skipping salt bridge analysis.")
         return defaultdict(list)

    sb_data = defaultdict(lambda: defaultdict(lambda: float('inf'))) # {(proc_res, prof_res): {frame: min_dist}}
    n_frames = len(universe.trajectory)

    # Wrap trajectory iteration with tqdm, applying the frame step
    trajectory_slice = universe.trajectory[::FRAME_STEP]
    n_analyzed_frames_sb = len(trajectory_slice)
    logger.info(f"Analyzing {n_analyzed_frames_sb} frames (step={FRAME_STEP})...")
    for ts in tqdm(trajectory_slice, total=n_analyzed_frames_sb, desc="Salt Bridges", unit="frame"):
        frame = ts.frame # Use 0-based frame index consistent with MDA

        # Case 1: PROC(+) <-> PROF(-)
        if has_proc_pos_prof_neg:
            # Calculate distances between all PROC(+) and PROF(-) atoms
            # Include box dimensions for periodic boundary conditions if available
            dist_matrix_1 = distances.distance_array(proc_pos_atoms.positions, prof_neg_atoms.positions, box=ts.dimensions)
            # Find atom pairs within the cutoff
            close_pairs_indices_1 = np.where(dist_matrix_1 <= dist_cutoff) # Returns tuple of arrays (row_indices, col_indices)

            # Process these close pairs
            for proc_idx, prof_idx in zip(*close_pairs_indices_1):
                proc_atom = proc_pos_atoms[proc_idx]
                prof_atom = prof_neg_atoms[prof_idx]
                # Define the residue pair consistently (PROC first)
                pair = (f"{proc_atom.resname.lower()}{proc_atom.resid}",
                        f"{prof_atom.resname.lower()}{prof_atom.resid}")
                dist = dist_matrix_1[proc_idx, prof_idx]
                # Update minimum distance for this pair in this frame
                if dist < sb_data[pair][frame]:
                    sb_data[pair][frame] = dist

        # Case 2: PROC(-) <-> PROF(+)
        if has_proc_neg_prof_pos:
            # Calculate distances between all PROC(-) and PROF(+) atoms
            dist_matrix_2 = distances.distance_array(proc_neg_atoms.positions, prof_pos_atoms.positions, box=ts.dimensions)
            # Find atom pairs within the cutoff
            close_pairs_indices_2 = np.where(dist_matrix_2 <= dist_cutoff)

            # Process these close pairs
            for proc_idx, prof_idx in zip(*close_pairs_indices_2):
                proc_atom = proc_neg_atoms[proc_idx]
                prof_atom = prof_pos_atoms[prof_idx]
                # Define the residue pair consistently (PROC first)
                pair = (f"{proc_atom.resname.lower()}{proc_atom.resid}",
                        f"{prof_atom.resname.lower()}{prof_atom.resid}")
                dist = dist_matrix_2[proc_idx, prof_idx]
                # Update minimum distance for this pair in this frame
                if dist < sb_data[pair][frame]:
                    sb_data[pair][frame] = dist

    # Convert to the final format: {pair: [(frame, min_dist)] sorted by frame}
    final_sb_data = defaultdict(list)
    for pair, frame_data in sb_data.items():
        final_sb_data[pair] = sorted(frame_data.items())

    logger.info(f"Processed Salt Bridge data for {len(final_sb_data)} unique PROC-PROF residue pairs.")
    return final_sb_data


def find_vdw_interactions(universe, proc_sel, prof_sel, vdw_atom_sel, dist_cutoff):
    logger.info("\nFinding VDW Interactions (Hydrophobic C-C)...")
    # Select the relevant C atoms in hydrophobic residues for each segment
    proc_vdw_atoms = universe.select_atoms(f"({proc_sel}) and ({vdw_atom_sel})")
    prof_vdw_atoms = universe.select_atoms(f"({prof_sel}) and ({vdw_atom_sel})")

    logger.info(f"Selected atoms - PROC(VDW): {len(proc_vdw_atoms)}, PROF(VDW): {len(prof_vdw_atoms)}")

    if len(proc_vdw_atoms) == 0 or len(prof_vdw_atoms) == 0:
        logger.warning("No potentially interacting VDW atom groups found. Skipping VDW analysis.")
        return defaultdict(list)

    vdw_data = defaultdict(lambda: defaultdict(lambda: float('inf'))) # {(proc_res, prof_res): {frame: min_dist}}
    n_frames = len(universe.trajectory)

    # Wrap trajectory iteration with tqdm, applying the frame step
    trajectory_slice = universe.trajectory[::FRAME_STEP]
    n_analyzed_frames_vdw = len(trajectory_slice)
    logger.info(f"Analyzing {n_analyzed_frames_vdw} frames (step={FRAME_STEP})...")
    for ts in tqdm(trajectory_slice, total=n_analyzed_frames_vdw, desc="VDW Interactions", unit="frame"):
        frame = ts.frame # 0-based index

        # Calculate distances between all selected PROC and PROF VDW atoms
        dist_matrix = distances.distance_array(proc_vdw_atoms.positions, prof_vdw_atoms.positions, box=ts.dimensions)
        # Find atom pairs within the cutoff
        close_pairs_indices = np.where(dist_matrix <= dist_cutoff)

        # Process these close pairs
        for proc_idx, prof_idx in zip(*close_pairs_indices):
            proc_atom = proc_vdw_atoms[proc_idx]
            prof_atom = prof_vdw_atoms[prof_idx]
            # Define the residue pair consistently (PROC first)
            pair = (f"{proc_atom.resname.lower()}{proc_atom.resid}",
                    f"{prof_atom.resname.lower()}{prof_atom.resid}")
            dist = dist_matrix[proc_idx, prof_idx]
            # Update minimum distance for this pair in this frame
            if dist < vdw_data[pair][frame]:
                vdw_data[pair][frame] = dist

    # Convert to the final format: {pair: [(frame, min_dist)] sorted by frame}
    final_vdw_data = defaultdict(list)
    for pair, frame_data in vdw_data.items():
        final_vdw_data[pair] = sorted(frame_data.items())

    logger.info(f"Processed VDW data for {len(final_vdw_data)} unique PROC-PROF residue pairs.")
    return final_vdw_data


def save_interaction_data_to_csv(interaction_data, interaction_type, output_dir, analyzed_frame_indices):
    """Saves interaction data to CSV files, ensuring only analyzed frames are present (padded with NaN)."""
    logger.info(f"\nSaving {interaction_type} data to CSV files in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    saved_files = 0
    skipped_empty = 0
    n_analyzed_frames = len(analyzed_frame_indices)

    # Wrap pairs iteration with tqdm for saving progress
    for (proc_res, prof_res), frame_dist_list in tqdm(interaction_data.items(), desc=f"Saving {interaction_type} CSVs", unit="pair"):
        # Filter out any potential infinite distances if they slipped through
        # This list already only contains analyzed frames from the steps above
        valid_frame_dist_list = [(f, d) for f, d in frame_dist_list if np.isfinite(d)]

        if not valid_frame_dist_list:
            skipped_empty += 1
            continue

        filename = f"{proc_res}_{prof_res}_{interaction_type}.csv"
        filepath = os.path.join(output_dir, filename)

        # Create a dictionary of distances for frames where the interaction was found
        # The keys (frames) should match the analyzed_frame_indices
        present_frames_dict = dict(valid_frame_dist_list)

        # Build the data list ONLY for the analyzed frames
        output_data = []
        for frame_index in analyzed_frame_indices:
            distance = present_frames_dict.get(frame_index, np.nan) # Get dist or NaN if not found for this analyzed frame
            output_data.append([frame_index, distance])

        # Create DataFrame
        df = pd.DataFrame(output_data, columns=['frame', 'distance'])

        # Check if frame count matches the number of analyzed frames
        if len(df) != n_analyzed_frames:
             logger.error(f"Frame count mismatch for {filename} after padding. Expected {n_analyzed_frames}, got {len(df)}. Skipping save.")
             continue

        try:
            # Save without header and index as requested
            df.to_csv(filepath, index=False, header=False, float_format='%.4f') # Format float precision
            saved_files += 1
        except Exception as e:
            logger.error(f"Error saving file {filepath}: {e}", exc_info=True)


    logger.info(f"Finished saving {interaction_type} data:")
    logger.info(f"- Saved {saved_files} files.")
    if skipped_empty > 0:
         logger.info(f"- Skipped {skipped_empty} pairs with no valid interactions.")


# --- Analysis and Plotting Functions (Adapted from User Input) ---

# === FUNCTION 1: Load all CSVs ===
def load_distance_data(csv_folder, n_analyzed_frames, analyzed_frame_indices):
    """
    Loads all interaction CSV files (_h-bond, _salt-bridge, _vdw) in the folder.
    Combines them, validates frame counts against analyzed frames, and adds identifiers.
    Expects n_analyzed_frames rows per file with correct sparse frame indices.
    """
    all_data = []
    processed_files = 0
    skipped_files = 0
    expected_rows = n_analyzed_frames
    expected_frame_set = set(analyzed_frame_indices)

    logger.info(f"\nLoading and combining CSV data from {csv_folder}")
    logger.info(f"Expecting {expected_rows} analyzed frames per file (Indices: {analyzed_frame_indices[:3]}...{analyzed_frame_indices[-1]}).")

    # Define expected suffixes and corresponding interaction types
    interaction_suffixes = {
        '_h-bond.csv': 'h-bond',
        '_salt-bridge.csv': 'salt-bridge',
        '_vdw.csv': 'vdw'
    }

    try:
        all_potential_files = [f for f in os.listdir(csv_folder) if any(f.lower().endswith(suffix) for suffix in interaction_suffixes)]
    except FileNotFoundError:
        logger.error(f"CSV input directory not found: {csv_folder}")
        sys.exit(1)

    logger.info(f"Found {len(all_potential_files)} potential interaction CSV files.")

    # Wrap file iteration with tqdm
    for file in tqdm(all_potential_files, desc="Loading CSVs", unit="file"):

        file_lower = file.lower()
        interaction_type = None
        suffix_used = None

        for suffix, itype in interaction_suffixes.items():
            if file_lower.endswith(suffix):
                interaction_type = itype
                suffix_used = suffix
                break

        if interaction_type:
            try:
                # Extract base name: procRes_profRes
                base_name = file[:-len(suffix_used)]
                if not base_name or len(base_name.split('_')) < 2: # Basic check
                    logger.warning(f"Could not parse residue pair from filename {file}. Skipping.")
                    skipped_files += 1
                    continue

                # Keep the full identifier including type for uniqueness
                pair_identifier = f"{base_name}_{interaction_type}"
                # logger.info(f"Processing {file} (Pair: {pair_identifier})...") # Verbose

                filepath = os.path.join(csv_folder, file)
                # Read CSV without headers and name columns explicitly
                df = pd.read_csv(filepath, header=None, names=['frame', 'distance'],
                               dtype={'frame': int, 'distance': float}) # Specify dtypes

                # --- Validation ---
                # 1. Check number of columns
                if len(df.columns) != 2:
                    logger.warning(f"{file} has unexpected number of columns ({len(df.columns)}). Skipping.")
                    skipped_files += 1
                    continue

                # 2. Check number of rows
                if len(df) != expected_rows:
                    logger.error(f"{file} has {len(df)} rows, expected {expected_rows}. This file might be corrupted or padding failed. Skipping.")
                    skipped_files += 1
                    continue # Skip file with wrong number of rows

                # 3. Check frame indices match the analyzed indices
                present_frame_set = set(df['frame']) # Get unique frames actually in the file
                if present_frame_set != expected_frame_set:
                     logger.error(f"Frame index mismatch for {file}. Expected indices like {list(expected_frame_set)[:5]}..., got {sorted(list(present_frame_set))[:5]}... Skipping.")
                     skipped_files += 1
                     continue

                # Add identifiers
                df["residue_pair_type"] = pair_identifier # e.g., arg59_glu76_salt-bridge
                df["interaction_type"] = interaction_type
                # Extract PROC and PROF res for potential later use
                parts = base_name.split('_')
                df["proc_res"] = parts[0]
                df["prof_res"] = parts[1]

                all_data.append(df)
                processed_files += 1

            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}", exc_info=True)
                # import traceback
                # traceback.print_exc() # Print stack trace for debugging
                skipped_files += 1
                continue
        # else: # Should not happen now due to the initial filtering
             # logger.warning(f"File {file} did not match expected suffixes during loading loop. Skipping.")
             # skipped_files += 1


    if not all_data:
        logger.critical("\nERROR: No valid CSV files were successfully processed. Cannot continue analysis.")
        logger.critical("Please check the CSV output directory, file formats, and logs.")
        sys.exit(1)

    logger.info(f"\nCSV Loading complete:")
    logger.info(f"- Successfully loaded and validated: {processed_files} files")
    logger.info(f"- Skipped or failed validation: {skipped_files} files")

    logger.info("Combining dataframes...")
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")

    # Final validation of frame count in combined data
    final_unique_frames = combined_df['frame'].nunique()
    if final_unique_frames != expected_rows:
        logger.warning(f"Final combined data has {final_unique_frames} unique frames, expected {expected_rows} analyzed frames.")
    else:
         logger.info(f"Final combined data contains expected number of unique analyzed frames ({final_unique_frames}).")

    return combined_df

# === FUNCTION 2: AIM 1 – Contact Frequency ===
def calculate_contact_frequencies(df):
    logger.info("\nCalculating Contact Frequencies...")
    if not all(['interaction_type' in df.columns, 'distance' in df.columns, 'residue_pair_type' in df.columns, 'frame' in df.columns]):
        raise ValueError("DataFrame missing required columns for contact frequency calculation.")
    if df.empty:
         logger.warning("Input DataFrame is empty for freq calc. Returning empty.")
         return pd.DataFrame(columns=['residue_pair_type', 'contact_frequency'])

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Define thresholds based on interaction type
    conditions = [
        (df['interaction_type'] == 'h-bond') & (df['distance'] <= HBOND_DIST_CUTOFF),
        (df['interaction_type'] == 'vdw') & (df['distance'] <= VDW_DIST_CUTOFF),
        (df['interaction_type'] == 'salt-bridge') & (df['distance'] <= SALT_BRIDGE_DIST_CUTOFF)
    ]
    choices = [True, True, True]

    # Default is False (no contact or NaN distance)
    df['contact'] = np.select(conditions, choices, default=False)

    # Ensure NaN distances do not count as contacts
    df.loc[df['distance'].isna(), 'contact'] = False

    # Calculate frequency per residue_pair_type
    total_frames = df['frame'].nunique()
    if total_frames == 0:
        logger.warning("No frames found in the dataframe for frequency calculation.")
        return pd.DataFrame(columns=['residue_pair_type', 'contact_frequency'])

    logger.info(f"Calculating frequencies over {total_frames} frames.")

    # Count frames where contact is True for each pair
    contact_counts = df[df['contact']].groupby('residue_pair_type').size()

    # Ensure all pairs are represented, even those with 0 contacts
    all_pairs = df['residue_pair_type'].unique()
    freq_df = pd.DataFrame({'residue_pair_type': all_pairs})
    freq_df = freq_df.merge(contact_counts.rename('contact_count'), on='residue_pair_type', how='left')
    freq_df['contact_count'] = freq_df['contact_count'].fillna(0).astype(int)

    # Calculate frequency
    freq_df['contact_frequency'] = freq_df['contact_count'] / total_frames

    # Sort by frequency
    freq_df = freq_df.sort_values(by='contact_frequency', ascending=False).reset_index(drop=True)

    logger.info(f"Calculated contact frequencies for {len(freq_df)} pairs.")
    return freq_df[['residue_pair_type', 'contact_frequency']] # Return relevant columns


# === FUNCTION 3: AIM 2 – Switching Events ===
def calculate_switching_events(df, analyzed_frame_indices):
    """
    Counts how often the set of active interactions (any type) changes between adjacent frames.
    Returns the total number of switches and a matrix encoding the highest priority interaction type.
    Priority: 3=Salt Bridge, 2=H-bond, 1=VDW, 0=None
    """
    logger.info("\nCalculating Switching Events and Interaction Type Matrix...")
    if not all(['interaction_type' in df.columns, 'distance' in df.columns, 'residue_pair_type' in df.columns, 'frame' in df.columns]):
        raise ValueError("DataFrame missing required columns for switching event calculation.")
    if df.empty:
        logger.warning("Input DataFrame is empty for switching calc. Returning 0, empty matrix.")
        return 0, pd.DataFrame()

    df = df.copy()

    # Determine contact status based on type-specific thresholds
    conditions = [
        (df['interaction_type'] == 'h-bond') & (df['distance'] <= HBOND_DIST_CUTOFF),
        (df['interaction_type'] == 'vdw') & (df['distance'] <= VDW_DIST_CUTOFF),
        (df['interaction_type'] == 'salt-bridge') & (df['distance'] <= SALT_BRIDGE_DIST_CUTOFF)
    ]
    choices = [True, True, True]
    df['contact'] = np.select(conditions, choices, default=False)
    df.loc[df['distance'].isna(), 'contact'] = False

    # --- Create Interaction Type Matrix --- #
    logger.info("Creating interaction type matrix (frames x pairs)...")
    # Define interaction type codes (higher value = higher priority)
    interaction_codes = {
        'salt-bridge': 3,
        'h-bond': 2,
        'vdw': 1
    }
    df['interaction_code'] = df['interaction_type'].map(interaction_codes).fillna(0).astype(int)

    # Create a temporary code column that is 0 if no contact
    df['active_code'] = df['interaction_code'] * df['contact']

    # Pivot table to get the *maximum* code for each frame/pair
    # This implements the priority: Salt (3) > Hbond (2) > VDW (1) > None (0)
    try:
        interaction_matrix = df.pivot_table(index='frame',
                                          columns='residue_pair_type',
                                          values='active_code',
                                          aggfunc='max', # Highest code wins -> priority
                                          fill_value=0) # Fill missing with 0 (no contact)
    except Exception as e:
         logger.error(f"Error creating interaction type pivot table: {e}", exc_info=True)
         return 0, pd.DataFrame() # Return empty on error

    # Ensure matrix has index for all analyzed frames
    interaction_matrix = interaction_matrix.reindex(analyzed_frame_indices, fill_value=0).sort_index()

    logger.info(f"Interaction type matrix created: {interaction_matrix.shape} (analyzed frames × pairs)")
    # Example: Show counts of different interaction types in the matrix (can be slow for large matrices)
    # logger.info(f"Interaction code counts: {np.unique(interaction_matrix.values, return_counts=True)}")

    # --- Calculate Switching Events using the boolean version --- #
    # We still need the boolean matrix for the original definition of switching
    logger.info("Creating boolean contact matrix for switching calculation...")
    try:
         contact_matrix = df.pivot_table(index='frame',
                                       columns='residue_pair_type',
                                       values='contact',
                                       aggfunc='any',
                                       fill_value=False)
    except Exception as e:
         logger.error(f"Error creating boolean contact pivot table: {e}", exc_info=True)
         # Return the interaction matrix we successfully created, but 0 switches
         return 0, interaction_matrix

    # Ensure boolean matrix also has full analyzed frame index
    contact_matrix = contact_matrix.reindex(analyzed_frame_indices, fill_value=False).sort_index()

    frames = contact_matrix.index.to_numpy()
    switches = 0

    if len(frames) < 2:
        logger.warning("Need at least 2 frames to calculate switches.")
        return 0, contact_matrix

    logger.info("Comparing adjacent analyzed frames for changes in contact sets...")
    # Get boolean matrix values
    matrix_values = contact_matrix.values
    # Frame indices in the matrix are now sparse (0, 10, 20...)
    matrix_frames = contact_matrix.index.to_numpy()

    # Wrap frame comparison loop with tqdm
    # Iterate up to the second-to-last analyzed frame index
    for i in tqdm(range(len(matrix_frames) - 1), desc="Comparing Frames", unit="step"):
        # Compare row i with row i+1 (representing frame f and frame f+STEP)
        if not np.array_equal(matrix_values[i], matrix_values[i+1]):
            switches += 1


    logger.info(f"Total switching events (changes in the set of active contacts): {switches}")
    # Return the *interaction_matrix* (with codes) for plotting, and the calculated switches
    return switches, interaction_matrix


# === FUNCTION 4: AIM 3 – Rebinding Events ===
def count_rebinding_events(df):
    logger.info("\nCounting Rebinding Events (Off -> On transitions per pair)...")
    if not all(['interaction_type' in df.columns, 'distance' in df.columns, 'residue_pair_type' in df.columns, 'frame' in df.columns]):
        raise ValueError("DataFrame missing required columns for rebinding event calculation.")
    if df.empty:
         logger.warning("Input DataFrame is empty for rebinding calc. Returning empty.")
         return pd.DataFrame(columns=['residue_pair_type', 'rebinding_count'])

    df = df.copy()

    # Determine contact status based on type-specific thresholds
    conditions = [
        (df['interaction_type'] == 'h-bond') & (df['distance'] <= HBOND_DIST_CUTOFF),
        (df['interaction_type'] == 'vdw') & (df['distance'] <= VDW_DIST_CUTOFF),
        (df['interaction_type'] == 'salt-bridge') & (df['distance'] <= SALT_BRIDGE_DIST_CUTOFF)
    ]
    choices = [True, True, True]
    df['contact'] = np.select(conditions, choices, default=False)
    df.loc[df['distance'].isna(), 'contact'] = False

    rebinding_counts = defaultdict(int)

    # Ensure data is sorted by frame within each group
    logger.info("Sorting data by pair and frame for rebinding calc...")
    df = df.sort_values(['residue_pair_type', 'frame'])

    # Calculate previous contact state using shift within each group
    logger.info("Calculating previous contact state...")
    df['prev_contact'] = df.groupby('residue_pair_type')['contact'].shift(1)

    # Identify rebinding events: previous was False (or NaN for frame 0) and current is True
    logger.info("Identifying rebinding events...")
    rebinding_mask = (~df['prev_contact'].fillna(False).astype(bool)) & df['contact']

    # Count events per group
    logger.info("Counting events per pair...")
    rebinding_series = df[rebinding_mask].groupby('residue_pair_type').size()

    # Convert to DataFrame
    if rebinding_series.empty:
         logger.info("No rebinding events detected.")
         return pd.DataFrame(columns=['residue_pair_type', 'rebinding_count'])


    rebinding_df = rebinding_series.reset_index(name='rebinding_count')
    rebinding_df = rebinding_df.sort_values('rebinding_count', ascending=False).reset_index(drop=True)

    logger.info(f"Counted rebinding events for {len(rebinding_df)} pairs.")
    return rebinding_df

# === FUNCTION 5: Find No Contact Frames ===
def find_no_contact_frames(contact_matrix):
    logger.info("\nFinding Frames with No PROC-PROF Contacts...")
    if contact_matrix.empty:
        logger.warning("Contact matrix is empty. Cannot find no-contact frames.")
        return np.array([]), []

    # Sum contacts across all pairs for each frame. If sum is 0, no contacts in that frame.
    contact_sums = contact_matrix.sum(axis=1)

    # Find frame indices where the sum is 0
    no_contact_frames_indices = contact_matrix.index[contact_sums == 0]
    no_contact_frames = no_contact_frames_indices.to_numpy() # Frame numbers

    logger.info(f"Found {len(no_contact_frames)} frames with no active contacts.")

    # Analyze consecutive gaps *within the analyzed frames*
    gaps = []
    if len(no_contact_frames) > 0:
        logger.info("Analyzing consecutive no-contact segments (based on analyzed frames)...")
        # Find where the difference between consecutive no-contact frame *indices* is greater than FRAME_STEP
        diffs = np.diff(no_contact_frames)
        # Indices where a new gap starts in the no_contact_frames array
        split_indices = np.where(diffs > FRAME_STEP)[0] + 1 # Check for jump > step

        # Split the no_contact_frames array into segments of consecutive *analyzed* frames
        consecutive_segments = np.split(no_contact_frames, split_indices)

        for segment in consecutive_segments:
            if len(segment) > 0: # Should always be true after split
                start_frame = segment[0]
                end_frame = segment[-1]
                # The size is the number of *analyzed* frames in the segment
                # To get duration, multiply by FRAME_STEP?
                gap_size_frames = len(segment)
                duration_estimate = gap_size_frames * FRAME_STEP
                gaps.append((start_frame, end_frame, gap_size_frames, duration_estimate))

        # Sort gaps by size (number of analyzed frames) in descending order
        gaps.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Found {len(gaps)} consecutive no-contact segments among analyzed frames.")
        # Optional: Print top gaps
        logger.info("Top 5 largest no-contact gaps (Analyzed Frames):")
        for i, (start, end, size, duration) in enumerate(gaps[:5]):
             logger.info(f"  {i+1}: Frames {start}-{end} ({size} analyzed frames, ~{duration} duration)")

    else:
        logger.info("No frames without contacts were found among analyzed frames.")


    return no_contact_frames, gaps


# === Plotting Functions (Adapted) ===
# Need adaptation for general pairs and interaction types

def plot_contact_frequencies(freq_df, output_dir):
    logger.info("\nPlotting Contact Frequencies...")
    if freq_df.empty:
        logger.warning("Frequency data is empty. Skipping plot.")
        return

    # Limit number of pairs shown to avoid overly cluttered plots
    max_pairs_to_plot = 75 # Adjust as needed
    if len(freq_df) > max_pairs_to_plot:
         logger.warning(f"Plotting only the top {max_pairs_to_plot} most frequent pairs out of {len(freq_df)}.")
         plot_df = freq_df.head(max_pairs_to_plot).copy()
    else:
         plot_df = freq_df.copy()


    # Prepare data for plotting
    plot_df['interaction_type'] = plot_df['residue_pair_type'].str.split('_').str[-1]
    # Simplify label: remove type suffix and replace last underscore with hyphen
    plot_df['pair_label'] = plot_df['residue_pair_type'].str.rsplit('_', n=1).str[0].str.replace('_', '-')

    # Create color mapping
    mode_colors = {
        'salt-bridge': '#D55E00',  # Orange-red
        'h-bond': '#56B4E9', # Light blue
        'vdw': '#006837',   # Dark green
        'other': '#999999'   # Gray for unexpected types
    }
    plot_df['color'] = plot_df['interaction_type'].apply(lambda x: mode_colors.get(x, mode_colors['other']))

    # Create figure
    fig_width = max(12, len(plot_df) * 0.2) # Adjust width based on number of bars
    plt.figure(figsize=(fig_width, 7))

    # Plot bars
    plt.bar(plot_df['pair_label'], plot_df['contact_frequency'],
            color=plot_df['color'], alpha=0.8)

    plt.xticks(rotation=80, ha='right', fontsize=max(6, 10 - len(plot_df)//10)) # Adjust fontsize slightly
    plt.ylabel('Contact Frequency')
    plt.xlabel('Residue Pair Interaction')
    plt.title(f'Top {len(plot_df)} Contact Frequencies between PROC and PROF')
    plt.ylim(0, max(1.0, plot_df['contact_frequency'].max() * 1.05)) # Ensure y-limit covers data, at least up to 1.0

    # Create legend handles manually for defined types present in the plotted data
    present_types = plot_df['interaction_type'].unique()
    legend_labels_map = {
        'salt-bridge': f'Salt Bridge (≤ {SALT_BRIDGE_DIST_CUTOFF} Å)',
        'h-bond': f'H-Bond (≤ {HBOND_DIST_CUTOFF} Å)',
        'vdw': f'VDW C-C (≤ {VDW_DIST_CUTOFF} Å)',
        'other': 'Other/Unknown'
    }
    active_handles = [plt.Rectangle((0,0),1,1, color=mode_colors[mode], alpha=0.8)
                     for mode in mode_colors if mode in present_types]
    active_texts = [legend_labels_map[mode] for mode in mode_colors if mode in present_types]

    if active_handles:
         plt.legend(active_handles, active_texts,
              title='Interaction Type & Cutoff',
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95]) # Adjust layout (left, bottom, right, top)

    output_plot = os.path.join(output_dir, "contact_frequencies_plot.png")
    try:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        logger.info(f"Contact frequencies plot saved to {output_plot}")
    except Exception as e:
        logger.error(f"Error saving contact frequencies plot: {e}", exc_info=True)
    plt.close()

    # Print summary (optional, print plotted data)
    logger.info("\nContact Frequencies Summary (Plotted Pairs):")
    logger.info(plot_df[['pair_label', 'contact_frequency']].to_string(index=False))

    # --- Add Plot for Top 8 Frequencies --- #
    logger.info("Plotting Top 8 Contact Frequencies...")
    top8_df = freq_df.head(8).copy() # Use original full freq_df
    if top8_df.empty:
        logger.warning("No data for top 8 frequency plot.")
        return # Skip if no data

    # Prepare labels and colors for top 8
    top8_df['interaction_type'] = top8_df['residue_pair_type'].str.split('_').str[-1]
    top8_df['pair_label'] = top8_df['residue_pair_type'].str.rsplit('_', n=1).str[0].str.replace('_', '-')
    top8_df['color'] = top8_df['interaction_type'].apply(lambda x: mode_colors.get(x, mode_colors['other']))

    # Create figure for top 8
    plt.figure(figsize=(10, 6))
    plt.bar(top8_df['pair_label'], top8_df['contact_frequency'],
            color=top8_df['color'], alpha=0.8)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylabel('Contact Frequency')
    plt.xlabel('Residue Pair Interaction')
    plt.title('Top 8 Contact Frequencies between PROC and PROF')
    plt.ylim(0, max(1.0, top8_df['contact_frequency'].max() * 1.05))

    # Create legend for top 8 plot
    present_types_top8 = top8_df['interaction_type'].unique()
    active_handles_top8 = [plt.Rectangle((0,0),1,1, color=mode_colors[mode], alpha=0.8)
                         for mode in mode_colors if mode in present_types_top8]
    active_texts_top8 = [legend_labels_map[mode] for mode in mode_colors if mode in present_types_top8]

    if active_handles_top8:
         plt.legend(active_handles_top8, active_texts_top8,
              title='Interaction Type & Cutoff',
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 0.85, 0.95]) # Adjust layout for legend

    output_plot_top8 = os.path.join(output_dir, "contact_frequencies_top8_plot.png")
    try:
        plt.savefig(output_plot_top8, dpi=300, bbox_inches='tight')
        logger.info(f"Top 8 Contact frequencies plot saved to {output_plot_top8}")
    except Exception as e:
        logger.error(f"Error saving top 8 contact frequencies plot: {e}", exc_info=True)
    plt.close()


def plot_rebinding_counts(rebinding_df, output_dir):
    logger.info("\nPlotting Rebinding Counts...")
    if rebinding_df.empty:
        logger.warning("Rebinding data is empty. Skipping plots.")
        return

    # Limit pairs shown
    max_pairs_to_plot = 75
    if len(rebinding_df) > max_pairs_to_plot:
         logger.warning(f"Plotting only the top {max_pairs_to_plot} pairs with most rebinding events out of {len(rebinding_df)}.")
         plot_df = rebinding_df.head(max_pairs_to_plot).copy()
    else:
         plot_df = rebinding_df.copy()

    # Prepare data
    plot_df['interaction_type'] = plot_df['residue_pair_type'].str.split('_').str[-1]
    # Simplify label
    plot_df['pair_label'] = plot_df['residue_pair_type'].str.rsplit('_', n=1).str[0].str.replace('_', '-')

    mode_colors = {
        'salt-bridge': '#D55E00', 'h-bond': '#56B4E9', 'vdw': '#006837', 'other': '#999999'
    }
    plot_df['color'] = plot_df['interaction_type'].apply(lambda x: mode_colors.get(x, mode_colors['other']))

    # Create legend elements
    present_types = plot_df['interaction_type'].unique()
    legend_labels_map = {'salt-bridge': 'Salt Bridge', 'h-bond': 'H-Bond', 'vdw': 'VDW', 'other': 'Other'}
    active_handles = [plt.Rectangle((0,0),1,1, color=mode_colors[mode], alpha=0.7)
                     for mode in mode_colors if mode in present_types]
    active_texts = [legend_labels_map[mode] for mode in mode_colors if mode in present_types]

    # --- Plot 1: With grid lines and counts ---
    fig_width = max(12, len(plot_df) * 0.2)
    fig1, ax1 = plt.subplots(figsize=(fig_width, 7))

    bars1 = ax1.bar(plot_df['pair_label'], plot_df['rebinding_count'], color=plot_df['color'], alpha=0.7)
    ax1.bar_label(bars1, fmt='%d', padding=3, fontsize=max(5, 8 - len(plot_df)//10)) # Add counts

    ax1.tick_params(axis='x', rotation=80, labelsize=max(6, 10 - len(plot_df)//10))
    ax1.set_ylabel('Number of Rebinding Events (Off→On)')
    ax1.set_xlabel('Residue Pair Interaction')
    ax1.set_title(f'Top {len(plot_df)} Rebinding Events (Counts & Grid)')
    if active_handles:
        ax1.legend(active_handles, active_texts, title='Interaction Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
    output_plot1 = os.path.join(output_dir, "rebinding_counts_plot_with_grid.png")
    try:
        plt.savefig(output_plot1, dpi=300, bbox_inches='tight')
        logger.info(f"Rebinding counts plot (with grid) saved to {output_plot1}")
    except Exception as e:
         logger.error(f"Error saving rebinding plot (grid): {e}", exc_info=True)
    plt.close(fig1)

    # --- Plot 2: Without grid lines ---
    fig2, ax2 = plt.subplots(figsize=(fig_width, 7))
    bars2 = ax2.bar(plot_df['pair_label'], plot_df['rebinding_count'], color=plot_df['color'], alpha=0.7)
    # Optional counts: ax2.bar_label(bars2, fmt='%d', padding=3, fontsize=max(5, 8 - len(plot_df)//10))

    ax2.tick_params(axis='x', rotation=80, labelsize=max(6, 10 - len(plot_df)//10))
    ax2.set_ylabel('Number of Rebinding Events (Off→On)')
    ax2.set_xlabel('Residue Pair Interaction')
    ax2.set_title(f'Top {len(plot_df)} Rebinding Events (No Grid)')
    if active_handles:
        ax2.legend(active_handles, active_texts, title='Interaction Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.grid(False)
    ax2.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
    output_plot2 = os.path.join(output_dir, "rebinding_counts_plot_without_grid.png")
    try:
        plt.savefig(output_plot2, dpi=300, bbox_inches='tight')
        logger.info(f"Rebinding counts plot (without grid) saved to {output_plot2}")
    except Exception as e:
         logger.error(f"Error saving rebinding plot (no grid): {e}", exc_info=True)
    plt.close(fig2)


def plot_contact_matrix(contact_matrix, output_dir):
    """Creates a heatmap of the interaction type matrix (frames x pairs)."""
    logger.info("\nPlotting Interaction Type Matrix Heatmap...")
    if contact_matrix.empty:
        logger.warning("Interaction type matrix is empty. Skipping heatmap plot.")
        return

    n_frames, n_pairs = contact_matrix.shape
    logger.info(f"Matrix dimensions: {n_frames} frames x {n_pairs} pairs.")

    # Limit pairs shown directly on heatmap if too many, plot subset or aggregate?
    # For now, let's just plot everything but manage labels.
    # max_pairs_for_labels = 100 # Show labels if fewer than this many pairs
    # show_xticklabels = n_pairs <= max_pairs_for_labels # Always show labels now

    # Adjust figure size dynamically
    fig_width = min(30, max(10, n_pairs * 0.1)) # Cap width at 30 inches
    fig_height = max(8, n_frames * 0.01) # Height scales with frames, min 8 inches
    fig_height = min(20, fig_height) # Cap height

    plt.figure(figsize=(fig_width, fig_height))

    # Define the colormap and normalization for interaction types
    # 0=None (White), 1=VDW (Green), 2=Hbond (Blue), 3=Salt (Orange)
    cmap_colors = [
        (1, 1, 1),       # 0: White (No contact)
        (0, 0.408, 0.216), # 1: Dark Green (VDW)
        (0.337, 0.706, 0.914), # 2: Light Blue (H-bond)
        (0.835, 0.369, 0)  # 3: Orange-Red (Salt Bridge)
    ]
    interaction_cmap = plt.cm.colors.ListedColormap(cmap_colors)
    norm = plt.cm.colors.BoundaryNorm([0, 1, 2, 3, 4], interaction_cmap.N) # Define boundaries for colors

    # Determine y-tick frequency based on number of frames
    ytick_frequency = max(1, n_frames // 20) # Aim for ~20 labels on y-axis

    logger.info(f"Plotting heatmap (Show X labels: {True}, Y tick freq: {ytick_frequency})...")
    try:
        ax = sns.heatmap(contact_matrix, # Use the matrix with interaction codes
                    cmap=interaction_cmap,
                    norm=norm, # Apply normalization for discrete colors
                    cbar=True, # Show color bar
                    cbar_kws={ # Customize color bar
                        'ticks': [0.5, 1.5, 2.5, 3.5], # Position ticks between boundaries
                        'label': 'Interaction Type Priority'
                        },
                    xticklabels=True, # Always attempt to show x labels
                    yticklabels=ytick_frequency # Show yticks at specified frequency
                   )
        # Set color bar tick labels
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(['None', 'VDW', 'H-bond', 'Salt Bridge'])

    except Exception as e:
         logger.error(f"Error during heatmap generation: {e}", exc_info=True)
         logger.warning("Plotting might fail for very large matrices due to memory constraints.")
         plt.close()
         return


    # if show_xticklabels: # Always true now, just apply the formatting
    ax.tick_params(axis='x', rotation=90, labelsize=max(4, 8 - n_pairs // 15)) # Adjust label size
    # else:
    #      plt.xlabel(f'{n_pairs} Interaction Pairs (Labels Omitted)')

    ax.tick_params(axis='y', labelsize=8)
    plt.ylabel('Frame Number')
    plt.title(f'Interaction Matrix ({n_frames} Frames x {n_pairs} Pairs)\nColor indicates highest priority interaction')

    # Remove the old manual legend
    # from matplotlib.patches import Patch
    # legend_elements = [Patch(facecolor='black', edgecolor='k', label='Contact Active'),
    #                    Patch(facecolor='white', edgecolor='k', label='No Contact')]
    # ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', title="Status", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.95, 0.96]) # Adjust layout slightly for legend/title

    output_plot = os.path.join(output_dir, "contact_matrix_heatmap.png")
    try:
        # Lower DPI for potentially very large matrices to manage file size/memory
        save_dpi = 150 if n_pairs > 200 or n_frames > 5000 else 300
        logger.info(f"Saving heatmap with DPI={save_dpi}...")
        plt.savefig(output_plot, dpi=save_dpi, bbox_inches='tight')
        logger.info(f"Contact matrix heatmap saved to {output_plot}")
    except Exception as e:
        logger.error(f"Error saving contact matrix heatmap: {e}", exc_info=True)
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    # Use logger instead of print for script start/end messages
    logger.info("=")
    logger.info("Starting Interaction Analysis Script")
    logger.info("=")

    # --- 1. Create Output Directories ---
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    logger.info(f"CSV Output Directory: {CSV_OUTPUT_DIR}")
    logger.info(f"Plot Output Directory: {PLOT_OUTPUT_DIR}")

    # --- 2. Find DCD Files ---
    dcd_files = find_all_dcd_files(DCD_BASE_DIR)
    if not dcd_files:
        logger.critical("\nERROR: No valid DCD files found. Exiting.")
        sys.exit(1)
    if not os.path.exists(PDB_FILE):
        logger.critical(f"\nERROR: PDB file not found: {PDB_FILE}. Exiting.")
        sys.exit(1)

    logger.info(f"\nUsing PDB: {PDB_FILE}")
    logger.info(f"Using {len(dcd_files)} DCD files found.")

    # --- Check if CSVs exist to potentially skip calculation ---
    # Flags to determine which calculations to run
    run_hbond_calc = True
    run_saltbridge_calc = True
    run_vdw_calc = True

    try:
        existing_files = os.listdir(CSV_OUTPUT_DIR)
        # Check each type individually
        if any(f.lower().endswith('_h-bond.csv') for f in existing_files):
            logger.info(f"Found existing H-bond CSV files in {CSV_OUTPUT_DIR}. Skipping H-bond calculation.")
            run_hbond_calc = False
        else:
            logger.warning(f"No existing H-bond CSV files found in {CSV_OUTPUT_DIR}. H-bond calculation will run.")

        if any(f.lower().endswith('_salt-bridge.csv') for f in existing_files):
            logger.info(f"Found existing Salt Bridge CSV files in {CSV_OUTPUT_DIR}. Skipping Salt Bridge calculation.")
            run_saltbridge_calc = False
        else:
            logger.warning(f"No existing Salt Bridge CSV files found in {CSV_OUTPUT_DIR}. Salt Bridge calculation will run.")

        if any(f.lower().endswith('_vdw.csv') for f in existing_files):
            logger.info(f"Found existing VDW CSV files in {CSV_OUTPUT_DIR}. Skipping VDW calculation.")
            run_vdw_calc = False
        else:
            logger.warning(f"No existing VDW CSV files found in {CSV_OUTPUT_DIR}. VDW calculation will run.")

    except FileNotFoundError:
        logger.info(f"CSV output directory {CSV_OUTPUT_DIR} not found. All calculations will run.")
        # Flags remain True

    # Determine if any calculation is needed
    needs_calculation = run_hbond_calc or run_saltbridge_calc or run_vdw_calc

    # Initialize variables that might be skipped
    n_frames = 0
    n_analyzed_frames = 0
    analyzed_frame_indices = []

    if needs_calculation:
        # --- 3. Load Universe ---
        logger.info(f"\nLoading Universe (needed for calculations)...")
        try:
            # Add guess_bonds=True to handle PDBs missing CONECT records
            u = mda.Universe(PDB_FILE, dcd_files, in_memory=False, guess_bonds=True)
            logger.info(f"Universe loaded successfully (with bond guessing):")
            logger.info(f"  {len(u.atoms)} atoms, {len(u.residues)} residues, {len(u.segments)} segments")
            n_frames = len(u.trajectory)
            logger.info(f"  {n_frames} frames")
        except Exception as e:
            logger.critical(f"\nERROR loading Universe: {e}", exc_info=True)
            logger.critical("Check PDB/DCD files and paths.")
            sys.exit(1)

        if n_frames == 0:
            logger.critical("\nERROR: Trajectory has 0 frames. Check DCD files.")
            sys.exit(1)

        analyzed_frame_indices = list(range(0, n_frames, FRAME_STEP))
        n_analyzed_frames = len(analyzed_frame_indices)
        logger.info(f"Total frames: {n_frames}. Analyzing {n_analyzed_frames} frames (every {FRAME_STEP}).")

        logger.info("\nSelecting PROC and PROF segments...")
        proc_group = u.select_atoms(PROC_SEL)
        prof_group = u.select_atoms(PROF_SEL)
        logger.info(f"Selected {len(proc_group)} atoms for PROC ('{PROC_SEL}')")
        logger.info(f"Selected {len(prof_group)} atoms for PROF ('{PROF_SEL}')")
        if len(proc_group) == 0 or len(prof_group) == 0:
            logger.critical("\nERROR: PROC or PROF selection resulted in 0 atoms. Check segids.")
            sys.exit(1)

        # --- 4. Run Interaction Analyses ---
        logger.info("\nStarting Interaction Calculations (for missing types)...")
        if run_hbond_calc:
            hbond_results = find_hydrogen_bonds(u, PROC_SEL, PROF_SEL, HBOND_DIST_CUTOFF)
        else:
            hbond_results = {} # Use empty dict if not calculated

        if run_saltbridge_calc:
            saltbridge_results = find_salt_bridges(u, PROC_SEL, PROF_SEL, POS_CHARGE_SEL, NEG_CHARGE_SEL, SALT_BRIDGE_DIST_CUTOFF)
        else:
            saltbridge_results = {}

        if run_vdw_calc:
            vdw_results = find_vdw_interactions(u, PROC_SEL, PROF_SEL, VDW_ATOM_SEL, VDW_DIST_CUTOFF)
        else:
            vdw_results = {}
        logger.info("\nFinished Interaction Calculations.")

        # --- 5. Save Raw Interaction Data to CSV ---
        logger.info("\nSaving Interaction Distances to CSV (for newly calculated types)...")
        if run_hbond_calc:
            save_interaction_data_to_csv(hbond_results, 'h-bond', CSV_OUTPUT_DIR, analyzed_frame_indices)
        if run_saltbridge_calc:
            save_interaction_data_to_csv(saltbridge_results, 'salt-bridge', CSV_OUTPUT_DIR, analyzed_frame_indices)
        if run_vdw_calc:
            save_interaction_data_to_csv(vdw_results, 'vdw', CSV_OUTPUT_DIR, analyzed_frame_indices)
        logger.info("\nFinished Saving Interaction CSVs.")

    else: # All calculations were skipped because all CSV types exist
        logger.info("\nAll calculations skipped. Inferring frame info from existing CSVs...")
        # Find one csv file to read frame info from
        first_csv = None
        try:
            csv_files = os.listdir(CSV_OUTPUT_DIR)
            for suffix in ['_h-bond.csv', '_salt-bridge.csv', '_vdw.csv']:
                for f in csv_files:
                    if f.lower().endswith(suffix):
                        first_csv = os.path.join(CSV_OUTPUT_DIR, f)
                        break
                if first_csv: break
        except FileNotFoundError:
            logger.critical(f"CSV directory {CSV_OUTPUT_DIR} disappeared after check? Exiting.")
            sys.exit(1)

        if not first_csv:
            logger.critical(f"Could not find any valid CSV files in {CSV_OUTPUT_DIR} despite skip check passing. Exiting.")
            sys.exit(1)

        try:
            # Read just the frame column
            temp_df = pd.read_csv(first_csv, header=None, usecols=[0], names=['frame'], dtype=int)
            analyzed_frame_indices = sorted(temp_df['frame'].unique().tolist())
            n_analyzed_frames = len(analyzed_frame_indices)
            # Estimate total frames and step (less reliable)
            if len(analyzed_frame_indices) > 1:
                 inferred_step = analyzed_frame_indices[1] - analyzed_frame_indices[0]
                 # Crude estimate of original total frames
                 n_frames = analyzed_frame_indices[-1] + inferred_step
                 logger.info(f"Inferred from {first_csv}: {n_analyzed_frames} analyzed frames with step ~{inferred_step}. Original total ~{n_frames}")
            else:
                 n_frames = n_analyzed_frames # Only one frame analyzed?
                 logger.info(f"Inferred from {first_csv}: {n_analyzed_frames} analyzed frames.")

        except Exception as e:
            logger.critical(f"Error reading frame info from {first_csv}: {e}. Cannot proceed.", exc_info=True)
            sys.exit(1)

    # Ensure we have valid frame info before loading
    if not analyzed_frame_indices or n_analyzed_frames == 0:
        logger.critical("Frame information (indices/count) is missing or invalid. Cannot load CSV data.")
        sys.exit(1)

    combined_data = load_distance_data(CSV_OUTPUT_DIR, n_analyzed_frames, analyzed_frame_indices)

    # Check if data loading was successful
    if combined_data.empty:
        logger.critical("Failed to load any data from CSVs. Exiting.")
        sys.exit(1)

    # --- 7. Perform Analyses ---
    logger.info("\nStarting Post-Analysis Calculations...")
    contact_freq_df = calculate_contact_frequencies(combined_data)
    # Pass the inferred/calculated analyzed_frame_indices to switching events
    switches, contact_matrix = calculate_switching_events(combined_data, analyzed_frame_indices)
    rebinding_df = count_rebinding_events(combined_data)

    no_contact_frames = np.array([])
    no_contact_gaps = []
    if not contact_matrix.empty:
        no_contact_frames, no_contact_gaps = find_no_contact_frames(contact_matrix)
    else:
        logger.warning("\nSkipping no-contact frame analysis because contact matrix is empty.")

    logger.info("\nFinished Post-Analysis Calculations.")

    # --- 8. Save No-Contact Frames Info ---
    logger.info(f"\nSaving no-contact frame information to {NO_CONTACT_FILE}...")
    try:
        with open(NO_CONTACT_FILE, 'w') as f:
            # ... (keep writing logic, maybe use logger inside if needed) ...
             f.write(f"Interaction Analysis Report\n")
             f.write(f"PDB: {PDB_FILE}\n")
             f.write(f"DCD Path: {DCD_BASE_DIR}\n")
             f.write(f"Total frames analyzed: {n_frames}\n")
             f.write(f"Frame Step: {FRAME_STEP} (Analyzed {n_analyzed_frames} frames)\n")
             f.write("-" * 30 + "\n")
             f.write(f"Total Switching Events (between analyzed frames): {switches}\n")
             f.write("-" * 30 + "\n")
             f.write(f"Frames with NO PROC-PROF Contacts: {len(no_contact_frames)}\n\n")

             if len(no_contact_frames) > 0:
                 f.write("Frame Indices (0-based) with No Contacts:\n")
                 # Write frames, wrap lines if very long list
                 frames_str = ','.join(map(str, no_contact_frames))
                 max_line_len = 100
                 for i in range(0, len(frames_str), max_line_len):
                      f.write(frames_str[i:i+max_line_len] + '\n')
                 f.write("\n")

                 f.write("Consecutive No-Contact Segments (based on analyzed frames):\n")
                 f.write("(Start Frame, End Frame, Num Analyzed Frames, Approx Duration Estimate)\n")
                 # Sort gaps by start frame for readability in the file
                 no_contact_gaps.sort(key=lambda x: x[0])
                 for start, end, size, duration in no_contact_gaps:
                     f.write(f"  Frames {start}-{end} ({size} analyzed frames, ~{duration} duration)\n")
             else:
                 f.write("No frames without contacts were found among analyzed frames.\n")
        logger.info("No-contact frame information saved.")
    except Exception as e:
        logger.error(f"ERROR writing no-contact file: {e}", exc_info=True)

    # --- 9. Generate Plots ---
    logger.info("\nGenerating Plots...")
    if not contact_freq_df.empty:
        plot_contact_frequencies(contact_freq_df, PLOT_OUTPUT_DIR)
    else:
        logger.info("Skipping contact frequency plot (no data).")

    if not rebinding_df.empty:
        plot_rebinding_counts(rebinding_df, PLOT_OUTPUT_DIR)
    else:
        logger.info("Skipping rebinding counts plot (no data).")

    if not contact_matrix.empty:
        plot_contact_matrix(contact_matrix, PLOT_OUTPUT_DIR)
    else:
        logger.info("Skipping contact matrix plot (matrix is empty).")

    logger.info("\n" + "=")
    logger.info("Interaction Analysis Script Finished Successfully.")
    logger.info("=")
