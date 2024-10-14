#!/bin/bash
#SBATCH --job-name=flux_inpainting_job   
#SBATCH --output=flux_inpainting_output_%j.txt  
#SBATCH --mail-type=ALL                   # Mail notifications 
#SBATCH --mail-user=alexis.allemand94@gmail.com
#SBATCH --partition=h100                  
#SBATCH --gres=gpu:1                      # Ask 1 GPU
#SBATCH --cpus-per-task=4                 # Ask 4 CPU
#SBATCH --mem=32G                         # Memory asked
#SBATCH --time=02:00:00                   # Time limit

module load gcc
source ~/env/bin/activate  

python run_flux_inpainting.py
