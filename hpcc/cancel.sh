#!/bin/bash

# Replace with your username
USER="sadeghib"

# pending jobs
job_ids=$(squeue -u "$USER" -t PD -h -o %A)

# all jobs
# job_ids=$(squeue -u "$USER" -h -o %A)

# Convert the job IDs to an array
job_ids_array=($job_ids)

# Cancel jobs in batches of 10
batch_size=10
for (( i=0; i<${#job_ids_array[@]}; i+=batch_size )); do
    batch=("${job_ids_array[@]:i:batch_size}")
    echo "Cancelling jobs: ${batch[@]}"
    scancel "${batch[@]}"
    sleep 1  # Add a short delay between batches to reduce load on the scheduler
done

echo "All jobs for user $USER have been cancelled."
