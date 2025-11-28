#!/bin/bash

JOB_DIR="jobs"
# Loop through all .sb files in the current directory
for sb_file in "$JOB_DIR"/*.sb
do
    echo "Submitting job script: $sb_file"
    sbatch "$sb_file"
done
