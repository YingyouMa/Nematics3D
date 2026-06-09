"""End-to-end example: MAT → smoothed director → visualization.

Runs the full pipeline on the sample MAT file in the data/ folder:
  1. Convert and smooth the MAT file  →  npy/ subfolder
  2. Visualize the result interactively

Quick-start for your own data
------------------------------
1. Replace MAT_PATH with the path to your own .mat file.
2. Replace OUTPUT_DIR with wherever you want the processed .npy files saved.
3. Update the frame number in visualize_frame() to match your file.
4. Run:  python example_full_pipeline.py
"""

import sys
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEMPLATE_DIR))

from mat_to_smoothed_director import process_mat_file
from visualize_frame import visualize_frame

# ── Step 1: preprocess ────────────────────────────────────────────────────────
# process_mat_file(mat_path, output_dir, sigma=1.45, *, is_overwrite=True)
#
#   mat_path    – Path to your MATLAB v7.3 / HDF5 .mat file containing the raw
#                 director field (the "DT_diag" file from the microscopy pipeline).
#
#   output_dir  – Root folder for processed outputs.  A sub-folder named after
#                 the .mat stem is created automatically, e.g.:
#                     npy/sample01_ds=35_dz=0.45_Frame0095_DT_diag/
#                 Inside it you will find:
#                     mask_xyz_largest_component.npy          – binary tissue mask
#                     mask_xyz_largest_component_bounds.json  – bounding-box metadata
#                     n_qfield_xyz_weighted_gaussian_sigma<X>um_xyz.npy – smoothed director
#                     metadata_weighted_gaussian_sigma<X>um_xyz.json    – voxel sizes etc.
#
#   sigma       – Gaussian smoothing radius in physical units (µm).  Default 1.45 µm.
#                 Smoothing is isotropic in physical space, so anisotropic voxel
#                 sizes are accounted for automatically.
#                 Increase sigma for noisier data; decrease for finer structures.
#
#   is_overwrite – If True (default), silently overwrite existing output files.
#                  Set to False to raise FileExistsError instead (useful in batch
#                  scripts where you want to avoid accidentally re-processing).
#
# Returns a dict with keys: mat, output_dir, mask, bounds, director, metadata
# – each value is the Path (as string) of the corresponding output file.

MAT_PATH = Path(__file__).parent / "data" / "sample01_ds=35_dz=0.45_Frame0095_DT_diag.mat"
OUTPUT_DIR = Path(__file__).parent / "npy"

result = process_mat_file(MAT_PATH, OUTPUT_DIR)
print("Preprocessing done:")
for key, val in result.items():
    print(f"  {key}: {val}")

# ── Step 2: visualize ─────────────────────────────────────────────────────────
# visualize_frame(frame_number, sigma=1.45, save_path=None, *, is_overwrite=True)
#
#   frame_number – Integer frame index, e.g. 95.  The function looks for a
#                  sub-folder under INPUT_ROOT whose name matches the glob
#                  "*Frame0095_DT_diag" (frame number zero-padded to 4 digits).
#                  The leading "*" is a wildcard, so "sample01_..._Frame0095_DT_diag"
#                  and "sample02_..._Frame0095_DT_diag" both match.
#                  If two folders under INPUT_ROOT share the same frame number
#                  (e.g. two samples mixed in one directory), an error is raised.
#                  → Keep each sample in its own OUTPUT_DIR to avoid conflicts.
#
#   sigma        – Must match the sigma used in process_mat_file above so the
#                  correct .npy file is located.  Default 1.45.
#
#   save_path    – If None (default), opens an interactive 3-D window.
#                  Pass a file path (e.g. "output/frame095.png") to render
#                  off-screen and save instead — useful for batch / headless runs.
#
#   is_overwrite – Same semantics as in process_mat_file: True (default) silently
#                  replaces an existing save_path; False raises FileExistsError.
#
# Returns the QFieldObject that was visualized (handy for further analysis).

# Point visualize_frame at the npy folder we just created
import visualize_frame as vf
vf.INPUT_ROOT = OUTPUT_DIR

visualize_frame(95)
