#!/bin/bash
#SBATCH --job-name=flux_inpainting_job   
#SBATCH --output=flux_inpainting_output_%j.txt  
#SBATCH --mail-type=ALL                   # Mail notifications 
#SBATCH --mail-user=alexis.allemand94@gmail.com
#SBATCH --partition=h100                  
#SBATCH --time=02:00:00                   # Time limit

module load gcc
source ~/env/bin/activate  

python run_flux_inpainting.py
