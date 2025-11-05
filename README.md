# Panoramic Segmentation

This repository contains code for applying Semantic Segmentation to 360degree Panoramic Streetview Imagery:


## What's inside

- `extract_lookaround_panorama.py` — fetches Apple Look Around panorama faces and stitches them into a single equirectangular JPEG.
- `360_semantic_segmentation.py` — segments a 360° panorama by projecting to a cubemap, segmenting each face, and reprojecting back to equirectangular (with an option to force the bottom face class) and outputting a segmentation map.
- `360segmentationPixelCounts.py` — runs the same segmentation and produces class coverage counts, including solid-angle–weighting appropriate for spherical imagery.

## Requirements

- Python 3.10+ (recommended)
- macOS, Linux, or Windows. Apple Silicon (MPS) and CUDA GPUs are supported if available; otherwise CPU is used.

Python packages (see `requirements.txt` for pinned versions):

- torch, torchvision
- transformers, huggingface-hub
- numpy, pillow, pillow-heif, matplotlib, pandas (for extras)
- streetlevel (Apple Look Around)
- pyproj, pyexiv2 (indirect, used by streetlevel utilities)
- py360convert

The first time you run a segmentation script, the Hugging Face model weights will download (~hundreds of MB) and be cached locally.

## Quickstart

Create and activate a virtual environment, then install dependencies.

```bash
# macOS / Linux (zsh)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

### 1) Download a panorama (optional)

`extract_lookaround_panorama.py` fetches Apple Look Around imagery near a latitude/longitude and stitches it to equirectangular.

Edit these variables near the top of the file before running:

- Coordinates: `lookaround.get_coverage_tile_by_latlon(lat, lon)`
- Desired panorama ID: set `target_id` if you know it, else the script uses the first pano on the tile. The panorama ID can be extracted from https://lookmap.eu.pythonanywhere.com .
- `zoom`: higher zoom increases output resolution (and download size)

Run it:

```bash
python extract_lookaround_panorama.py
```

Output: `"{pano_id}_{zoom}.jpg"` (equirectangular panorama).

Notes:

- The script uses `streetlevel` and internally handles authentication. It requires internet access.
- HEIC/HEIF faces are decoded via `pillow-heif`; the script registers the opener for you.
- Stitching to equirectangular requires PyTorch.

### 2) Segment a 360° panorama

`360_semantic_segmentation.py` uses the ADE20K Mask2Former model (`facebook/mask2former-swin-small-ade-semantic`). It splits the panorama into 6 cube faces, segments each face, and reprojects the colored labels back to equirectangular.

Edit these variables near the top:

- `image_name`: base filename (without extension) of your panorama, e.g., `"my_pano"` for `my_pano.jpg`
- `OVERRIDE_BOTTOM_CLASS_ID`: set to an integer ADE20K class ID to force the bottom face (face index 5) to a fixed class (this can be useful because the segmentation doesn't usually do well on bottom face of cube where images only show ground), or set to `None` for normal segmentation

Run it:

```bash
python 360_semantic_segmentation.py
```

Output: `"{image_name}_360_segmented.png"` with ADE20K color palette.

Acceleration: The script auto-selects the best device available in this order: Apple MPS (Apple Silicon), CUDA GPU, then CPU. You’ll see a console message indicating the device in use.

### 3) Get per-class coverage (with solid-angle weighting)

`360segmentationPixelCounts.py` runs segmentation and produces a CSV of class coverage. It accounts for spherical distortion by weighting cubemap pixels by their solid angle, giving a more accurate percentage of the scene per class.

Edit these variables near the top:

- `image_name`: base filename (without extension), e.g., `"my_pano"`
- `FACE_RES`: cube face resolution; lower values are faster but coarser (e.g., 256 or 512)
- `FORCE_FACE_INDEX` and `FORCE_CLASS_ID`: optionally force one face to a fixed class (set index to `None` to disable)

Run it:

```bash
python 360segmentationPixelCounts.py
```

Output: `"{image_name}_panorama_class_counts.csv"` with columns:

- `class_id`
- `pixel_count_raw`, `percent_raw` (simple pixel counts across all faces)
- `angular_weighted_count`, `percent` (weighted by solid angle; sums to ~100%)

Tip: If you only need counts (not the colored visualization), this script skips generating a full equirect label map for speed.

## ADE20K labels and colors

The scripts use the ADE20K taxonomy provided by the model. Class IDs are integers starting at 0. The segmentation visualization applies the standard ADE20K color palette.

- Model card: https://huggingface.co/facebook/mask2former-swin-small-ade-semantic
- ADE20K dataset info: https://groups.csail.mit.edu/vision/datasets/ADE20K/

If you need human-readable class names, consult the model card or ADE20K label mapping resources and join by `class_id` from the CSV output.


## Acknowledgements

- Mask2Former: Cheng et al., 2022 https://doi.org/10.48550/arXiv.2112.01527
- ADE20K: Zhou et al., 2016 https://doi.org/10.48550/arXiv.1608.05442
- `py360convert` for cube/equirectangular projection utilities https://github.com/sunset1995/py360convert 
- `streetlevel` for Apple Look Around programmatic access: https://github.com/sk-zk/streetlevel 

