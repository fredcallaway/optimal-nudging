#!/usr/bin/env bash
#SBATCH --job-name=validate
#SBATCH --output=out/%A_%a
#SBATCH --array=1-1000
#SBATCH --time=60
#SBATCH --mem-per-cpu=500
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=flc2@princeton.edu

module load julia/1.5.0  
echo runnnig job `expr $1 + $SLURM_ARRAY_TASK_ID`
julia validate_job.jl `expr $1 + $SLURM_ARRAY_TASK_ID`