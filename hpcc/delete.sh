#!/bin/bash

# Set the size threshold in bytes (670MB = 670 * 1024 * 1024 bytes)
threshold=$((670 * 1024 * 1024))

# Find and delete directories smaller than the threshold
find ./results/2Pulse-Characterization/ -type d | while read -r dir; do
    # Calculate the directory size in bytes
    dir_size=$(du -sb "$dir" | awk '{print $1}')
    
    # Check if the directory size is smaller than the threshold
    if [ "$dir_size" -lt "$threshold" ]; then
        echo "Deleting directory: $dir (Size: $dir_size bytes)"
        rm -rf "$dir"
    fi
done
