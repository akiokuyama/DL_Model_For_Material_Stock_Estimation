#!/bin/bash
#SBATCH --job-name=FA_ViT_Della_90_5_5    # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=8          # cpu-cores per task (adjust according to your specific task requirements)
#SBATCH --mem-per-cpu=8G            # memory per cpu-core (adjust according to your specific task requirements)
#SBATCH --gres=gpu:4                 # number of GPUs per node (you can use all 4 GPUs per node)
#SBATCH --time=5:00:00               # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin            # send email when job begins
#SBATCH --mail-type=end              # send email when job ends
#SBATCH --mail-type=fail             # send email if job fails
#SBATCH --mail-user=


export MASTER_PORT=$(get_free_port)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load anaconda3/2023.9
conda activate torch-env

srun python FA_ViT_Della_90_5_5.py
