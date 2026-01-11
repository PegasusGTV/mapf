#!/bin/bash
# Run MAPF demos on multiple maps using conda environment

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mapf

# Change to mapf directory
cd "$(dirname "$0")"

# Run the script
PYTHONPATH=. python scripts/run_mapf_demos.py \
    --maps warehouse-20-40-10-2-2 room-64-64-8 random-64-64-10 maze-128-128-2 ht_mansion_n \
    --k 50 \
    --fps 5 \
    --stride 1 \
    --motion 4 \
    --results_dir results \
    --max_timesteps 300

echo "Done! Check the results/ directory for GIFs and path files."

