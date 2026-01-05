# Geometry_CHARMM_Simulations_Analysis_PPI
Summary:
These scripts form a post-processing toolkit for your geometry/λ simulations: they merge and de-overlap dvdl.dat and fl.dat across restarts into clean pandas tables (csv/parquet), convert λ to RMSD and generate plots (RMSD vs timestep/frame and FEL-style surfaces), identify energy minima and analyze/extract trajectories/frames (MDAnalysis-based), and separately process residue–residue distance time series into CSVs and higher-level contact metrics (frequencies, switching, rebinding, and multi-site comparisons), plus utilities for mapping frames and making movies.

These scripts are written for Python and assume you have:
python>=3.8
pandas, numpy, matplotlib
plotly (interactive minima picking)
MDAnalysis (trajectory reading + frame extraction)
scikit-learn (some clustering in dcd_analysis.py)
ffmpeg (only if you use process_movie.py)

Disclaimer: This pipeline assumes your simulations were run with CHARMM (or a compatible setup) that outputs dvdl.dat, fl.dat, and trajectory files (*.dcd) along with a structure/topology file (*.psf or *.pdb) in the expected directory layout.

These scripts assume each run directory (the main <system> and any nested restart subdirectories) contains dvdl.dat, fl.dat, and one or more *.dcd files, plus a structure/topology file (*.psf or *.pdb); restarts are organized as subfolders under the main run path (e.g., hosts/<host>/<system>/<restart1>/<restart2>/...).

Order of running scripts:
1. python merge_restarts.py  hosts/<host>/<system>/<restart1>/
2. python process_geo.py     hosts/<host>/<system>/<restart1>/
3. python rmsd_values.py     hosts/<host>/<system>/<restart1>/
4. python find_frames.py     hosts/<host>/<system>/<restart1>/
5. python rmsd_vs_frame.py   hosts/<host>/<system>/<restart1>/   # optional
6. python dcd_analysis.py    hosts/<host>/<system>/<restart1>/   # edit minima_ranges first
after VMD distance extraction:
7. python process_distance_data.py /path/to/minima_data_folder 1
8. python geometry_analysis_all_sites.py # edit USER SETUP first OR
8. python geometry_analysis.py # edit USER SETUP first

Script roles:

Restart merging / cleanup
geometry.py – core helpers to validate geometry dirs and merge/trim overlaps for dvdl/fl
merge_restarts.py – runs the merge and writes merged dvdl/fl to csv/parquet
overlap.sh – makes overlap slices (debug helper)
de-overlap-all.sh – batch-runs merging over many geometry folders
process_geo.py – plotting/processing runner using the merged outputs (legacy/alt driver)

RMSD-as-order-parameter + plotting
rmsd_values.py – adds RMSD from λ mapping; writes *_rmsd.csv/parquet + plots
find_frames.py – adds frame_number to produce dvdl_rmsd_frame.csv/parquet
rmsd_vs_frame.py – plots RMSD vs frame from dvdl_rmsd_frame.csv

Minima finding + trajectory analysis
find_minima.py – finds wells/minima and extracts/organizes minima frames/mini-trajectories
dcd_analysis.py – heavier MDAnalysis-based analysis/summary (plots, clustering, etc.)
create_final_plots.py – regenerates final summary plots (memory-optimized)

Interaction distance processing + contact analysis
find_interactions.py – scans trajectory for interactions by distance cutoffs; writes per-interaction outputs
process_distance_data.py – converts site_* distance files → CSV and makes grouped plots
generic_distance_plotter.py – simpler “one folder” converter/plotter for distance files
geometry_analysis.py – single-site contact frequency/switching/rebinding analysis from distance CSVs
geometry_analysis_all_sites.py – same, but multi-site + co-occurrence across sites

Utilities / visualization
frame_mapping.py – maps an extracted frame back to the original trajectory frame index
process_movie.py – turns .ppm frame images into a high-quality MP4 via ffmpeg
