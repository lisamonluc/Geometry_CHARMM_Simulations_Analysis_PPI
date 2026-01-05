# Geometry_CHARMM_Simulations_Analysis_PPI

# What this repo does:
This repo is a set of scripts I use to post-process CHARMM geometry free energy sampling simulations where λ maps to RMSD (the selected order parameter). The main workflow merges `dvdl.dat` and `fl.dat` across restarts into clean CSV/Parquet tables, converts λ to a RMSD metric and makes plots (RMSD versus time or frame number, free energy landscapes with respect to RMSD), then identifies minima and extracts the frames in each basin into new DCD files (via MDAnalysis). A separate branch takes residue–residue distance time series (often exported from VMD) and turns them into contact behavior plots like contact fraction, contact matrices, switching, and rebinding (single-site or multi-site).

# Requirements:
- Python 3.8+
- pandas, numpy, matplotlib
- plotly (interactive minima picking)
- MDAnalysis (trajectory loading + frame extraction)
- scikit-learn (used in `dcd_analysis.py`)
- ffmpeg (only if you use `process_movie.py`)

# Disclaimer:
These scripts assume you ran CHARMM (or a compatible workflow) that outputs `dvdl.dat`, `fl.dat`, and trajectory files (`*.dcd`), along with a structure/topology file (`*.psf` or `*.pdb`).

# Expected directory layout:
Each run directory (the main run and any nested restart folders) should contain:
- `dvdl.dat`
- `fl.dat`
- one `*.dcd` file
- a structure/topology file: `*.psf` or `*.pdb`

Restarts are expected to be nested under the main folder, e.g.: `hosts/<host>/<system>/<restart1>/<restart2>/<restart3>/`

# Recommended run order:
1. Merge restart outputs:
- python merge_restarts.py hosts/<host>/<system>/<restart1>/
- python process_geo.py hosts/<host>/<system>/<restart1>/

2. Convert λ→RMSD and make FEL versus RMSD plots:
- python rmsd_values.py hosts/<host>/<system>/<restart1>/

3. Find frame numbers within minima:
- python find_frames.py hosts/<host>/<system>/<restart1>/
  
4. Plot RMSD versus frame (optional):
- python rmsd_vs_frame.py hosts/<host>/<system>/<restart1>/
  
5. Extract minima trajectories and run bond analysis:
- python dcd_analysis.py hosts/<host>/<system>/<restart1>/
  
6. After VMD distance extraction of bonds, convert distance text files to CSV and make distance-vs-frame plots:
- python process_distance_data.py /path/to/minima_data_folder 1

7. Contact behavior plots (pick all sites or one site at a time):
- python geometry_analysis_all_sites.py or single-site:python geometry_analysis.py
