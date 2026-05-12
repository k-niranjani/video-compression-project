#!/bin/bash
#SBATCH --job-name=extract_orig
#SBATCH --output=/export/fs06/ahallur1/cholec80/logs/extract_orig_%j.out
#SBATCH --error=/export/fs06/ahallur1/cholec80/logs/extract_orig_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

module purge
module load conda
conda deactivate
conda activate /home/ahallur1/miniconda3/envs/transsv

python /export/fs06/ahallur1/cholec80/TMRNet/code/extract_frames_original.py