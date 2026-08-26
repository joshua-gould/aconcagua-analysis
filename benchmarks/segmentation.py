"""
Segmentation Benchmark

Compares cell/nuclei segmentation methods:
- Cellpose 3 (cyto3 model)
- Cellpose 4 (CPSAM model - requires separate environment)
- StarDist (2D_versatile_fluo)

Metrics: runtime, memory, nuclei/cell counts, retention rates, pipeline efficiency.

Usage:
    python segmentation.py                     # Run all available methods
    python segmentation.py --method cellpose   # Run Cellpose (v3, cyto3)
    python segmentation.py --method cellpose4  # Run Cellpose 4 (CPSAM)
    python segmentation.py --method stardist   # Run only StarDist
"""

import argparse
import gc
import glob
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from tifffile import imread, imwrite
from tqdm import tqdm

# Add brieflow to path (relative to benchmarks directory)
BENCHMARKS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BENCHMARKS_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT / "brieflow" / "workflow"))


# Parse command line arguments first to determine which imports are needed
def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation benchmarks")
    parser.add_argument(
        "--method",
        type=str,
        choices=["cellpose", "cellpose4", "stardist", "all", "scallops"],
        default="all",
        help="Segmentation method to benchmark (default: all available)",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only regenerate plots from existing results",
    )
    return parser.parse_args()


# Parse args at module level so imports can be conditional
args = parse_args()
SELECTED_METHOD = args.method

# Conditionally import segmentation methods based on --method argument
CELLPOSE_AVAILABLE = False
CELLPOSE4_AVAILABLE = False
STARDIST_AVAILABLE = False

segment_cellpose = None
segment_stardist = None

if SELECTED_METHOD in ["cellpose", "all"]:
    try:
        from lib.shared.segment_cellpose import segment_cellpose

        CELLPOSE_AVAILABLE = True
        print("Cellpose 3 loaded successfully")
    except ImportError as e:
        print(f"Warning: Cellpose not available ({e}). Skipping Cellpose benchmark.")

if SELECTED_METHOD in ["cellpose4", "all"]:
    try:
        from lib.shared.segment_cellpose import segment_cellpose

        CELLPOSE4_AVAILABLE = True
        print("Cellpose 4 (CPSAM) loaded successfully")
    except ImportError as e:
        print(
            f"Warning: Cellpose 4 not available ({e}). Skipping Cellpose 4 benchmark."
        )

if SELECTED_METHOD in ["stardist", "all"]:
    try:
        from lib.shared.segment_stardist import segment_stardist

        STARDIST_AVAILABLE = True
        print("StarDist loaded successfully")
    except ImportError as e:
        print(f"Warning: StarDist not available ({e}). Skipping StarDist benchmark.")

random.seed(42)

# Configure paths (relative to project root)
SOURCE_DIR = PROJECT_ROOT / "analysis" / "brieflow_output" / "phenotype" / "images"
OUTPUT_DIR = BENCHMARKS_DIR / "results" / "segmentation"
ALIGNED_DIR = OUTPUT_DIR / "aligned"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
ALIGNED_DIR.mkdir(exist_ok=True, parents=True)

# Create subdirectories for each segmentation method
for method in ["cellpose", "cellpose4", "stardist"]:
    (OUTPUT_DIR / method).mkdir(exist_ok=True)

# Parameters for segmentation methods (from config.yml)
CELLPOSE_PARAMS = {
    "dapi_index": 0,
    "cyto_index": 3,
    "nuclei_diameter": 42.684959745981,
    "cell_diameter": 68.05913211132304,
    "cellpose_model": "cyto3",
    "nuclei_flow_threshold": 0.4,
    "nuclei_cellprob_threshold": 0.0,
    "cell_flow_threshold": 1,
    "cell_cellprob_threshold": 0,
    "reconcile": "contained_in_cells",
    "return_counts": True,
    "gpu": False,
    "segment_cells": True,
}

# Cellpose 4 with CPSAM model (requires cellpose 4.x)
CELLPOSE4_PARAMS = {
    "dapi_index": 0,
    "cyto_index": 3,
    "nuclei_diameter": 42.684959745981,
    "cell_diameter": 68.05913211132304,
    "cellpose_model": "cpsam",  # CPSAM model in Cellpose 4.x
    "nuclei_flow_threshold": 0.4,
    "nuclei_cellprob_threshold": 0.0,
    "cell_flow_threshold": 1,
    "cell_cellprob_threshold": 0,
    "reconcile": "contained_in_cells",
    "return_counts": True,
    "gpu": False,
    "segment_cells": True,
    "helper_index": 1,  # TUBULIN channel to help with CPSAM segmentation
}

STARDIST_PARAMS = {
    "dapi_index": 0,
    "cyto_index": 3,
    "stardist_model": "2D_versatile_fluo",
    "nuclei_prob_threshold": 0.479071,
    "nuclei_nms_threshold": 0.3,
    "cell_prob_threshold": 0.479071,
    "cell_nms_threshold": 0.3,
    "reconcile": "contained_in_cells",
    "return_counts": True,
    "gpu": False,
    "segment_cells": True,
}


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def _segment_scallops(image, plate, well, tile):
    if STARDIST_AVAILABLE:
        print("  Running scallops...")

        # Measure memory before GC to establish true baseline
        gc.collect()
        start_mem = get_memory_usage()
        start_time = time.time()
        import xarray as xr
        from scallops.segmentation.propagation import segment_cells_propagation
        from scallops.segmentation.stardist import segment_nuclei_stardist
        image = xr.DataArray(image, dims=("c", "y", "x"))
        nuclei_sd = segment_nuclei_stardist(image, nuclei_channel=STARDIST_PARAMS["dapi_index"])

        cells_sd, threshold = segment_cells_propagation(
            image=image,
            nuclei=nuclei_sd,
            threshold="Li",
            cyto_channel=STARDIST_PARAMS["cyto_index"],
            nuclei_channel=STARDIST_PARAMS["dapi_index"])
        counts = {}
        counts["final_nuclei"] = len(np.unique(nuclei_sd)) - 1
        counts["final_cells"] = len(np.unique(cells_sd)) - 1
        counts_df = pd.DataFrame([counts])
        scallops_time = time.time() - start_time
        end_mem = get_memory_usage()
        mem_usage = max(0, end_mem - start_mem)

        # Save segmentation results
        nuclei_out = (
                OUTPUT_DIR / "scallops" / f"P-{plate}_W-{well}_T-{tile}_nuclei.tiff"
        )
        cells_out = (
                OUTPUT_DIR / "scallops" / f"P-{plate}_W-{well}_T-{tile}_cells.tiff"
        )
        segmentation_stats_out = (
                OUTPUT_DIR
                / "scallops"
                / f"P-{plate}_W-{well}_T-{tile}_segmentation_stats.tsv"
        )

        # Add info to the beginning of the counts_df (metadata)
        counts_df.insert(0, "method", "scallops")
        counts_df.insert(1, "plate", plate)
        counts_df.insert(2, "well", well)
        counts_df.insert(3, "tile", tile)

        # Add metrics to the end of the counts_df (performance metrics)
        counts_df["runtime_seconds"] = scallops_time
        counts_df["memory_mb"] = mem_usage
        counts_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        imwrite(nuclei_out, nuclei_sd.astype(np.uint16))
        imwrite(cells_out, cells_sd.astype(np.uint16))
        counts_df.to_csv(segmentation_stats_out, index=False, sep="\t")

        # Clean up to reduce memory usage
        del nuclei_sd, cells_sd, counts_df
        gc.collect()



def prepare_test_images():
    """Find one aligned image per plate randomly and copy to benchmark directory."""
    # Pattern to match aligned image files
    pattern = str(SOURCE_DIR / "*__aligned.tiff")
    all_images = glob.glob(pattern)
    print(f"Found {len(all_images)} total aligned images")

    # Extract plate info from filenames
    images_by_plate = {}
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        if img_name.startswith("P-"):
            plate = img_name.split("_")[0].replace("P-", "")
            if plate not in images_by_plate:
                images_by_plate[plate] = []
            images_by_plate[plate].append(img_path)

    # Select a random image from each plate
    selected_images = []
    for plate, images in images_by_plate.items():
        if images:
            # Choose a random image from this plate
            random_image = random.choice(images)
            dst_path = ALIGNED_DIR / os.path.basename(random_image)
            print(
                f"Copying random tile from plate {plate}: {os.path.basename(random_image)}"
            )
            if not dst_path.exists():
                os.system(f"cp {random_image} {dst_path}")
            selected_images.append(dst_path)
        else:
            print(f"No images found for plate {plate}")

    print(f"\nCopied {len(selected_images)} images to {ALIGNED_DIR}")
    return selected_images


def collect_segmentation_stats():
    """Collect all segmentation stats into a single benchmark results file."""
    all_stats = []

    for method in ["cellpose", "cellpose4", "stardist"]:
        stat_files = list((OUTPUT_DIR / method).glob("*_segmentation_stats.tsv"))
        for file in stat_files:
            try:
                stats = pd.read_csv(file, sep="\t")
                all_stats.append(stats)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if all_stats:
        # Combine all stats into a single DataFrame
        benchmark_results = pd.concat(all_stats, ignore_index=True)
        # Save to CSV
        benchmark_results.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)
        print(f"Saved benchmark results to {OUTPUT_DIR / 'benchmark_results.csv'}")
        return benchmark_results
    else:
        print("No segmentation stats found.")
        return None


def benchmark_segmentation_methods():
    """Run benchmark tests on all three segmentation methods."""
    # Get test images
    image_paths = list(ALIGNED_DIR.glob("*.tiff"))
    # if not image_paths:
    #     print("No images found. Preparing test images...")
    #     image_paths = prepare_test_images()

    print(f"Found {len(image_paths)} aligned images for benchmarking")

    # Process each image
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_name = img_path.name

        # Extract plate, well, tile info from filename
        if "P-" in img_name and "W-" in img_name and "T-" in img_name:
            plate = img_name.split("_")[0].replace("P-", "")
            well = img_name.split("_")[1].replace("W-", "")
            tile = img_name.split("_")[2].replace("T-", "")
        else:
            # Default values if pattern doesn't match
            plate = "unknown"
            well = "unknown"
            tile = "unknown"

        print(f"\nProcessing plate {plate}, well {well}, tile {tile}")

        # Load image - these are already aligned
        try:
            image = imread(img_path)
            print(f"  Image shape: {image.shape}")
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue

        # Benchmark Cellpose
        if CELLPOSE_AVAILABLE:
            print("  Running Cellpose...")
            try:
                # Measure memory before GC to establish true baseline
                gc.collect()
                start_mem = get_memory_usage()
                start_time = time.time()

                # Get segmentation with reconciliation
                nuclei_cp, cells_cp, counts_df = segment_cellpose(
                    data=image,
                    dapi_index=CELLPOSE_PARAMS["dapi_index"],
                    cyto_index=CELLPOSE_PARAMS["cyto_index"],
                    nuclei_diameter=CELLPOSE_PARAMS["nuclei_diameter"],
                    cell_diameter=CELLPOSE_PARAMS["cell_diameter"],
                    cyto_model=CELLPOSE_PARAMS["cellpose_model"],
                    cellpose_kwargs=dict(
                        nuclei_flow_threshold=CELLPOSE_PARAMS["nuclei_flow_threshold"],
                        nuclei_cellprob_threshold=CELLPOSE_PARAMS[
                            "nuclei_cellprob_threshold"
                        ],
                        cell_flow_threshold=CELLPOSE_PARAMS["cell_flow_threshold"],
                        cell_cellprob_threshold=CELLPOSE_PARAMS[
                            "cell_cellprob_threshold"
                        ],
                    ),
                    reconcile=CELLPOSE_PARAMS["reconcile"],
                    return_counts=CELLPOSE_PARAMS["return_counts"],
                    gpu=CELLPOSE_PARAMS["gpu"],
                    cells=CELLPOSE_PARAMS["segment_cells"],
                )

                cellpose_time = time.time() - start_time
                end_mem = get_memory_usage()
                mem_usage = max(0, end_mem - start_mem)

                # Save segmentation results
                nuclei_out = (
                        OUTPUT_DIR / "cellpose" / f"P-{plate}_W-{well}_T-{tile}_nuclei.tiff"
                )
                cells_out = (
                        OUTPUT_DIR / "cellpose" / f"P-{plate}_W-{well}_T-{tile}_cells.tiff"
                )
                segmentation_stats_out = (
                        OUTPUT_DIR
                        / "cellpose"
                        / f"P-{plate}_W-{well}_T-{tile}_segmentation_stats.tsv"
                )

                # Add info to the beginning of the counts_df (metadata)
                counts_df.insert(0, "method", "cellpose")
                counts_df.insert(1, "plate", plate)
                counts_df.insert(2, "well", well)
                counts_df.insert(3, "tile", tile)

                # Add metrics to the end of the counts_df (performance metrics)
                counts_df["runtime_seconds"] = cellpose_time
                counts_df["memory_mb"] = mem_usage
                counts_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                imwrite(nuclei_out, nuclei_cp.astype(np.uint16))
                imwrite(cells_out, cells_cp.astype(np.uint16))
                counts_df.to_csv(segmentation_stats_out, index=False, sep="\t")

                # Clean up to reduce memory usage
                del nuclei_cp, cells_cp, counts_df
                gc.collect()

            except Exception as e:
                print(f"  Error running Cellpose: {e}")

        # Benchmark StarDist
        if STARDIST_AVAILABLE:
            print("  Running StarDist...")
            try:
                # Measure memory before GC to establish true baseline
                gc.collect()
                start_mem = get_memory_usage()
                start_time = time.time()

                # Get segmentation with reconciliation
                nuclei_sd, cells_sd, counts_df = segment_stardist(
                    data=image,
                    dapi_index=STARDIST_PARAMS["dapi_index"],
                    cyto_index=STARDIST_PARAMS["cyto_index"],
                    model_type=STARDIST_PARAMS["stardist_model"],
                    stardist_kwargs=dict(
                        nuclei_prob_threshold=STARDIST_PARAMS["nuclei_prob_threshold"],
                        nuclei_nms_threshold=STARDIST_PARAMS["nuclei_nms_threshold"],
                        cell_prob_threshold=STARDIST_PARAMS["cell_prob_threshold"],
                        cell_nms_threshold=STARDIST_PARAMS["cell_nms_threshold"],
                    ),
                    reconcile=STARDIST_PARAMS["reconcile"],
                    return_counts=STARDIST_PARAMS["return_counts"],
                    gpu=STARDIST_PARAMS["gpu"],
                    cells=STARDIST_PARAMS["segment_cells"],
                )

                stardist_time = time.time() - start_time
                end_mem = get_memory_usage()
                mem_usage = max(0, end_mem - start_mem)

                # Save segmentation results
                nuclei_out = (
                        OUTPUT_DIR / "stardist" / f"P-{plate}_W-{well}_T-{tile}_nuclei.tiff"
                )
                cells_out = (
                        OUTPUT_DIR / "stardist" / f"P-{plate}_W-{well}_T-{tile}_cells.tiff"
                )
                segmentation_stats_out = (
                        OUTPUT_DIR
                        / "stardist"
                        / f"P-{plate}_W-{well}_T-{tile}_segmentation_stats.tsv"
                )

                # Add info to the beginning of the counts_df (metadata)
                counts_df.insert(0, "method", "stardist")
                counts_df.insert(1, "plate", plate)
                counts_df.insert(2, "well", well)
                counts_df.insert(3, "tile", tile)

                # Add metrics to the end of the counts_df (performance metrics)
                counts_df["runtime_seconds"] = stardist_time
                counts_df["memory_mb"] = mem_usage
                counts_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                imwrite(nuclei_out, nuclei_sd.astype(np.uint16))
                imwrite(cells_out, cells_sd.astype(np.uint16))
                counts_df.to_csv(segmentation_stats_out, index=False, sep="\t")

                # Clean up to reduce memory usage
                del nuclei_sd, cells_sd, counts_df
                gc.collect()

            except Exception as e:
                print(f"  Error running StarDist: {e}")

        # Benchmark Cellpose 4 (CPSAM)
        if CELLPOSE4_AVAILABLE:
            print("  Running Cellpose 4 (CPSAM)...")
            try:
                # Measure memory before GC to establish true baseline
                gc.collect()
                start_mem = get_memory_usage()
                start_time = time.time()

                # Get segmentation with reconciliation using CPSAM model
                nuclei_cp4, cells_cp4, counts_df = segment_cellpose(
                    data=image,
                    dapi_index=CELLPOSE4_PARAMS["dapi_index"],
                    cyto_index=CELLPOSE4_PARAMS["cyto_index"],
                    nuclei_diameter=CELLPOSE4_PARAMS["nuclei_diameter"],
                    cell_diameter=CELLPOSE4_PARAMS["cell_diameter"],
                    cyto_model=CELLPOSE4_PARAMS["cellpose_model"],
                    cellpose_kwargs=dict(
                        nuclei_flow_threshold=CELLPOSE4_PARAMS["nuclei_flow_threshold"],
                        nuclei_cellprob_threshold=CELLPOSE4_PARAMS[
                            "nuclei_cellprob_threshold"
                        ],
                        cell_flow_threshold=CELLPOSE4_PARAMS["cell_flow_threshold"],
                        cell_cellprob_threshold=CELLPOSE4_PARAMS[
                            "cell_cellprob_threshold"
                        ],
                    ),
                    helper_index=CELLPOSE4_PARAMS["helper_index"],
                    reconcile=CELLPOSE4_PARAMS["reconcile"],
                    return_counts=CELLPOSE4_PARAMS["return_counts"],
                    gpu=CELLPOSE4_PARAMS["gpu"],
                    cells=CELLPOSE4_PARAMS["segment_cells"],
                )

                cellpose4_time = time.time() - start_time
                end_mem = get_memory_usage()
                mem_usage = max(0, end_mem - start_mem)

                # Save segmentation results
                nuclei_out = (
                        OUTPUT_DIR
                        / "cellpose4"
                        / f"P-{plate}_W-{well}_T-{tile}_nuclei.tiff"
                )
                cells_out = (
                        OUTPUT_DIR / "cellpose4" / f"P-{plate}_W-{well}_T-{tile}_cells.tiff"
                )
                segmentation_stats_out = (
                        OUTPUT_DIR
                        / "cellpose4"
                        / f"P-{plate}_W-{well}_T-{tile}_segmentation_stats.tsv"
                )

                # Add info to the beginning of the counts_df (metadata)
                counts_df.insert(0, "method", "cellpose4")
                counts_df.insert(1, "plate", plate)
                counts_df.insert(2, "well", well)
                counts_df.insert(3, "tile", tile)

                # Add metrics to the end of the counts_df (performance metrics)
                counts_df["runtime_seconds"] = cellpose4_time
                counts_df["memory_mb"] = mem_usage
                counts_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                imwrite(nuclei_out, nuclei_cp4.astype(np.uint16))
                imwrite(cells_out, cells_cp4.astype(np.uint16))
                counts_df.to_csv(segmentation_stats_out, index=False, sep="\t")

                # Clean up to reduce memory usage
                del nuclei_cp4, cells_cp4, counts_df
                gc.collect()

            except Exception as e:
                print(f"  Error running Cellpose 4 (CPSAM): {e}")
        if STARDIST_AVAILABLE:
            _segment_scallops(image, plate, well, tile)

        # Clean up image to reduce memory usage
        del image
        gc.collect()

    # Collect all segmentation stats into benchmark results
    results_df = collect_segmentation_stats()

    # Generate summary statistics
    if results_df is not None and len(results_df) > 0:
        # Generate summary visualizations
        generate_benchmark_summary(results_df)
    else:
        print("No results generated. Check for errors.")

    return results_df


def generate_benchmark_summary(results_df):
    """Generate summary statistics and plots from benchmark results"""
    import numpy as np
    from plot_style import setup_plot_style, box_strip, FIGSIZE, COLORS, save_figure, print_summary_table

    # Apply consistent plot styling
    setup_plot_style()

    # Use consistent method colors
    method_order = sorted(results_df["method"].unique())
    palette = {m: COLORS.get(m, "#7f7f7f") for m in method_order}

    # Sort methods alphabetically
    results_df = results_df.sort_values("method")

    # Calculate retention rates if they don't exist already
    if "nuclei_retention" not in results_df.columns:
        # Calculate retention rates, avoiding division by zero
        results_df["nuclei_retention"] = np.where(
            results_df["initial_nuclei"] > 0,
            results_df["final_nuclei"] / results_df["initial_nuclei"] * 100,
            0,
        )

    if "cell_retention" not in results_df.columns:
        results_df["cell_retention"] = np.where(
            results_df["initial_cells"] > 0,
            results_df["final_cells"] / results_df["initial_cells"] * 100,
            0,
        )

    # Calculate summary statistics by method
    summary = (
        results_df.groupby("method")
        .agg(
            {
                "initial_nuclei": ["mean", "std", "min", "max"],
                "initial_cells": ["mean", "std", "min", "max"],
                "after_edge_removal_nuclei": ["mean", "std", "min", "max"],
                "after_edge_removal_cells": ["mean", "std", "min", "max"],
                "final_nuclei": ["mean", "std", "min", "max"],
                "final_cells": ["mean", "std", "min", "max"],
                "nuclei_retention": ["mean", "std", "min", "max"],
                "cell_retention": ["mean", "std", "min", "max"],
                "runtime_seconds": ["mean", "std", "min", "max"],
                "memory_mb": ["mean", "std", "min", "max"],
            }
        )
        .reset_index()
    )

    # Save summary to file
    summary.to_csv(OUTPUT_DIR / "benchmark_summary.csv")

    # Runtime comparison (square)
    fig, ax = plt.subplots(figsize=FIGSIZE["square"])
    box_strip(ax, results_df, "method", "runtime_seconds", palette, method_order,
              ylabel="Runtime (seconds)", title="Runtime per Tile")
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "runtime_comparison.png")
    plt.close()

    # Memory usage comparison (square)
    fig, ax = plt.subplots(figsize=FIGSIZE["square"])
    box_strip(ax, results_df, "method", "memory_mb", palette, method_order,
              ylabel="Memory Usage (MB)", title="Memory Usage per Tile")
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "memory_comparison.png")
    plt.close()

    # Count comparison (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fig.suptitle("Cell and Nuclei Counts by Method", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.35)
    for ax, col, title in zip(
            axes.flat,
            ["initial_nuclei", "initial_cells", "final_cells", "runtime_seconds"],
            ["Initial Nuclei", "Initial Cells", "Final Cells", "Runtime (s)"],
    ):
        box_strip(ax, results_df, "method", col, palette, method_order,
                  ylabel="Count per Tile" if "nuclei" in col or "cells" in col else "Seconds",
                  title=title, fmt="int" if "cells" in col or "nuclei" in col else "auto")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_ylabel(ax.get_ylabel(), fontsize=13)
    # Shared legend at the bottom
    handles = [plt.matplotlib.patches.Patch(color=palette[m], label=m, alpha=0.6)
               for m in method_order]
    fig.legend(handles=handles, loc="lower center", ncol=len(method_order),
               fontsize=13, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, OUTPUT_DIR / "count_comparison.png")
    plt.close()

    # Retention rate plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Retention Rates by Method", fontweight="bold", y=0.98)
    box_strip(ax1, results_df, "method", "nuclei_retention", palette, method_order,
              ylabel="Retention Rate (%)", title="Nuclei Retention", fmt="pct",
              ylim=(0, 105))
    box_strip(ax2, results_df, "method", "cell_retention", palette, method_order,
              ylabel="Retention Rate (%)", title="Cell Retention", fmt="pct",
              ylim=(0, 105))
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "retention_rates.png")
    plt.close()

    # Create retention by stage plots
    # Calculate the differences for the stacked bars
    results_df["edge_removed_nuclei"] = (
            results_df["initial_nuclei"] - results_df["after_edge_removal_nuclei"]
    )
    results_df["edge_removed_cells"] = (
            results_df["initial_cells"] - results_df["after_edge_removal_cells"]
    )
    results_df["reconciled_nuclei"] = (
            results_df["after_edge_removal_nuclei"] - results_df["final_nuclei"]
    )
    results_df["reconciled_cells"] = (
            results_df["after_edge_removal_cells"] - results_df["final_cells"]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Processing Pipeline by Method", fontweight="bold", y=0.98)

    # Use viridis colormap for stacked bars
    stack_colors = plt.cm.viridis(np.linspace(0.1, 0.9, 3))

    # Nuclei pipeline with cleaner appearance
    data = results_df.groupby("method")[
        ["final_nuclei", "reconciled_nuclei", "edge_removed_nuclei"]
    ].mean()
    data = data.reindex(sorted(data.index))  # Sort alphabetically
    data.plot(kind="bar", stacked=True, ax=ax1, alpha=0.8, color=stack_colors)
    ax1.set_title("Nuclei Processing Pipeline")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Method")
    ax1.legend(
        ["Final Nuclei", "Lost in Reconciliation", "Edge Removed"],
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
    )

    # Cells pipeline with cleaner appearance
    data = results_df.groupby("method")[
        ["final_cells", "reconciled_cells", "edge_removed_cells"]
    ].mean()
    data = data.reindex(sorted(data.index))  # Sort alphabetically
    data.plot(kind="bar", stacked=True, ax=ax2, alpha=0.8, color=stack_colors)
    ax2.set_title("Cell Processing Pipeline")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Method")
    ax2.legend(
        ["Final Cells", "Lost in Reconciliation", "Edge Removed"],
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "processing_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print publication-ready summary tables
    pct_fmt = lambda m, lo, hi: f"{m:.1f}% [{lo:.1f}–{hi:.1f}%]"
    print_summary_table(
        "Segmentation — Runtime & Resources",
        results_df, "method",
        [("runtime_seconds", "Runtime (s)"), ("memory_mb", "Memory (MB)")],
    )
    print_summary_table(
        "Segmentation — Cell & Nuclei Counts",
        results_df, "method",
        [("initial_nuclei", "Initial Nuclei"), ("initial_cells", "Initial Cells"),
         ("final_nuclei", "Final Nuclei"), ("final_cells", "Final Cells")],
    )
    print_summary_table(
        "Segmentation — Retention Rates",
        results_df, "method",
        [("nuclei_retention", "Nuclei Ret. (%)"), ("cell_retention", "Cell Ret. (%)")],
        fmt_map={"nuclei_retention": pct_fmt, "cell_retention": pct_fmt},
    )


if __name__ == "__main__":
    if args.plots_only:
        # Load existing results and regenerate plots
        results_path = OUTPUT_DIR / "benchmark_results.csv"
        if results_path.exists():
            print("Regenerating plots from existing results...")
            results_df = pd.read_csv(results_path)
            generate_benchmark_summary(results_df)
        else:
            print(f"No results found at {results_path}. Run benchmark first.")
    else:
        benchmark_segmentation_methods()
