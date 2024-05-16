#!/bin/bash
#SBATCH --job-name=BT_60_20_20    # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes # increase after parallel
#SBATCH --constraint=a100            # specify to use A100 GPU
#SBATCH --cpus-per-task=8          # cpu-cores per task (adjust according to your specific task requirements)
#SBATCH --mem-per-cpu=20G            # memory per cpu-core (adjust according to your specific task requirements)
#SBATCH --gres=gpu:1                 # number of GPUs per node (you can use all 4 GPUs per node)
#SBATCH --time=5:00:00               # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin            # send email when job begins
#SBATCH --mail-type=end              # send email when job ends
#SBATCH --mail-type=fail             # send email if job fails
#SBATCH --mail-user=ao3526@princeton.edu


module purge
module load anaconda3/2023.9
conda activate torch-env2

srun python Eval_BT_ViT_Della_60_20_20.py