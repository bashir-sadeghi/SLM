import os
def generate_sb_script(job_name, init_epochs, control_epochs, max_k, time, gpus):
    sb_script = f"""#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time={time}             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                   # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                 # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=1          # number of CPUs (or cores) per task (same as -c)
#SBATCH --gpus={gpus}
#SBATCH --mem-per-cpu=32G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name={job_name}    # you can give your job a name for easier identification (same as -J)
#SBATCH --output=./jobs_outputs/{job_name}.out    # output file to write the log to

########## Command Lines for Job Running ##########

export PATH=$PATH:/mnt/research/SDAQ/sadeghib/miniconda3/bin
module purge
module load Conda/3

conda activate base

cd /mnt/research/SDAQ/sadeghib/frib

echo python main.py --args exps/sean/ae2.ini --init-epochs={init_epochs} --control-epochs={control_epochs} --max-k={max_k}

date

srun python main.py --args exps/sean/ae2.ini --init-epochs={init_epochs} --control-epochs={control_epochs} --max-k={max_k}

date
"""
    return sb_script

# Fixed parameters
time = "2:59:00"
gpus = "v100:1"


init_epochs_set = [70, 80, 90, 100, 110, 120, 130]
control_epochs_set = [70, 80, 90, 100, 110, 120, 130]
max_k_set = [4, 5, 6, 7, 8, 9, 10, 12]

try:
    if not os.path.exists('./jobs'):
        os.makedirs('./jobs')
except:
    pass


# Iterate over parameter sets
for init_epochs in init_epochs_set:
    for control_epochs in control_epochs_set:
        for max_k in max_k_set:
            # Generate a unique job name based on the parameters
            job_name = f"job_init{init_epochs}_control{control_epochs}_maxk{max_k}"
            
            # Generate the script content
            sb_script_content = generate_sb_script(job_name, init_epochs, control_epochs, max_k, time, gpus)
            
            # Write the script to a file
            with open(f"./jobs/{job_name}.sb", "w") as file:
                file.write(sb_script_content)
            
            print(f"./Jobs/'{job_name}.sb' generated successfully.")