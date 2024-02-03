#!/bin/bash

DIRECTORY="/home/orl/temp0130/halfcheetah-medexp-ablation"
PYTHON_SCRIPT="src/scripts/envtest/get_novelty_and_diversity.py"

num_of_samples=100000
device=cpu
dataset=halfcheetah-medium-v2

# Loop through each file in the directory
for file in "$DIRECTORY"/*
do
    for ratio in 1.0
    do
        # Check if the current item is a file
        if [ -f "$file" ]; then
            echo "Processing $file..." &
            # Run Python script with the file as an argument
            python src/scripts/envtest/get_novelty_and_diversity.py --dataset=$dataset --datapath=$file --num_of_samples=$num_of_samples --device=$device --ratio=$ratio
        fi
    done
done
wait

echo "All files processed."