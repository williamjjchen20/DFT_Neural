#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=logs/output_%j.log     # Log file (%j = job ID)
#SBATCH --error=logs/error_%j.log       # Error file
#SBATCH --partition=TWIG-GPU
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wchen5@andrew.cmu.edu

#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
#export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

module load cuda/12.4.0

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate myenv

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi 

echo Training Started!
#srun python model.py 
srun python validate.py
echo Training Finished!
