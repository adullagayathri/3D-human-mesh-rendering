3D Human Mesh Reconstruction & Motion Comparison (OSX)
Overview

This repository allows you to reconstruct 3D human meshes from video using SMPL-X, smooth motion sequences, and render output meshes.

Everything can be run directly from the notebook, which calls the scripts (demo.py, render.py, smooth.py, etc.) where needed. You don’t need to manually run scripts from the command line unless you want to.

Getting Started
1. Download the repository

Download or clone the repo:

git clone https://github.com/yourusername/osx-3d-reconstruction.git
cd osx-3d-reconstruction

2. Open the Notebook

Open OSX_notebook.ipynb in Jupyter or Colab.

3. Install Dependencies

Run the first cell in the notebook to install all necessary Python packages:

!pip install -r requirements.txt


This ensures all dependencies for demo.py, render_demo.py, smooth_render.py, gvsmooth_demo.py are installed.

Running the Experiment

Video Input

Upload a video or use a sample video provided.

The notebook will call demo.py internally to process the video into .npz latent representations.

Optional Precomputed Input

If you already have .npz files, the notebook can load them directly instead of running full reconstruction.

Smooth Motion

The notebook calls gvsmooth_demo.py automatically to clean sequences.

Render Meshes

The notebook calls smooth_render.py or render_demo.py to create output videos from reconstructed meshes.

File Summary
File	Purpose
OSX_notebook.ipynb	Main notebook. Runs full pipeline using scripts.
demo.py	Converts video frames to 3D meshes and saves .npz.
render_demo.py	Renders meshes to video for visualization.
gv_smooth.py	Smooths motion sequences.
smooth.py	Utility for smoothing 3D meshes.
render.py	Utility for rendering meshes.
Workflow in Notebook

Install dependencies

Upload video or .npz file

Run reconstruction (demo.py)

Smooth motion (gv_smooth.py)

Render output (render_demo.py / smooth_render.py)

Each step is a separate notebook cell. You can run all cells sequentially to reproduce the full experiment.

Notes

Some OSX models are large (~4.7GB). If using precomputed .npz, downloading models can be skipped.

The notebook integrates all scripts so users do not need to run Python scripts manually.

Recommended environment: Python 3.9+
