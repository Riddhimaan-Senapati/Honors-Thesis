#!/bin/bash

#SBATCH -p gpu  # Submit job to the gpu partition
#SBATCH -t 48:00:00     # Set maximum job time to 2 days 
#SBATCH --mail-type=TIME_LIMIT_80 # Email me if time exceed 80 percent of the allocated time or 38.4 hours
#SBATCH --gpus=1        # Request 1 GPU
#SBATCH --output=gpu_job_UTILITY_prepend_none.out  # Name the output file with the job ID and experiment choices

# Pull Ollama docker image
apptainer pull docker://ollama/ollama

# Setup Ollama
apptainer instance run --env "OLLAMA_HOST=127.0.0.1:11434" --nv ollama_latest.sif  ollama

# Setup location of Ollama model variable
export OLLAMA_MODELS=/home/rsenapati_umass_edu/.ollama/models/

# Download the Ollama model
#apptainer  run --env "OLLAMA_HOST=127.0.0.1:11434" --nv ollama_latest.sif  pull qwen3:0.6b

# Activate the virtual environment
source /work/pi_allan_umass_edu/rsenapati/.venv/bin/activate

# run the python script
python /work/pi_allan_umass_edu/rsenapati/run_experiments.py --prompt_type UTILITY --attack_type prepend --mitigation_type none --output_dir /work/pi_allan_umass_edu/rsenapati/data

