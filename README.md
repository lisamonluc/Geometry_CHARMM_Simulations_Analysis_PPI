# Geometry_CHARMM_Simulations_Analysis_PPI

# Summary:
# This repository contains a set of scripts I use to post-process CHARMM geometry free energy sampling MD simulations where the order parameter (λ) is represented by RMSD. The main workflow merges `dvdl.dat` and `fl.dat` across restarts into clean CSV/Parquet tables, converts λ to a RMSD metric, and makes plots (RMSD versus time or frame number, free energy landscapes with respect to RMSD), then identifies energy minima and extracts the frames in each basin into new DCD files. Afterwards, a separate script identifies popular contact points between protein partners whose distance can be tracked via the residue–residue distance time series exported from VMD. You can also manually identify these popular contact points via VMD. It converts them into contact behavior plots, including contact fraction, contact matrices, switching, and rebinding (single-site or multi-site).

# Requirements:
# - Python 3.8+
# - pandas, numpy, matplotlib
# - plotly (interactive minima picking)
# - MDAnalysis (trajectory loading + frame extraction)
# - scikit-learn (used in `dcd_analysis.py`)
# - ffmpeg (only if you use `process_movie.py`)

# Disclaimer:
# These scripts assume you ran CHARMM (or a compatible workflow) that outputs `dvdl.dat`, `fl.dat`, and trajectory files (`*.dcd`), along with a structure/topology file (`*.psf` or `*.pdb`).

# Expected directory layout:
# Each run directory (the main run and any nested restart folders) should contain:
# - `dvdl.dat`
# - `fl.dat`
# - `*.dcd`
# - a structure/topology file: `*.psf` or `*.pdb`

# Restarts are expected to be nested under the main folder, e.g.: `hosts/<host>/<system>/<restart1>/<restart2>/<restart3>/` This should be the paths called for steps 1-5.
# Place all scripts under working directory and create "hosts" directory within working directory

# Recommended run order:
# 1. Merge restart outputs:
# - python merge_restarts.py </path/to/trajectory_files>
# - python process_geo.py </path/to/trajectory_files>

# 2. Make sure to adjust slope value in script, convert λ to RMSD values and make Free Energy Landscape versus RMSD plots:
# - python rmsd_values.py </path/to/trajectory_files>

# 3. Find frame numbers within energy minima:
# - python find_frames.py </path/to/trajectory_files>
  
# 4. Plot RMSD versus frame (optional):
# - python rmsd_vs_frame.py </path/to/trajectory_files>
  
# 5. Extract minima trajectories and run bond analysis:
# - python dcd_analysis.py </path/to/trajectory_files>
  
# 6. After VMD distance extraction of bonds, convert distance text files to CSV and make distance-vs-frame plots:
# - python process_distance_data.py </path/to/minima_data_folder_1>

# 7. Contact behavior plots (pick all sites or one site at a time):
# - python geometry_analysis_all_sites.py or single-site: python geometry_analysis.py 

