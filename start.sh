#!/usr/bin/bash

# Pull Ollama docker image
apptainer pull docker://ollama/ollama

# Setup Ollama
apptainer instance run --env "OLLAMA_HOST=127.0.0.1:11434" --nv ollama_latest.sif  ollama

# Setup location of Ollama model
export OLLAMA_MODELS=/work/pi_allan_umass_edu/.ollama/models/

# Download the Ollama model
ollama pull gpt-oss:20b

# Activate the virtual environment
source /work/pi_allan_umass_edu/rsenapati/.venv/bin/activate

# run the python script
python /work/pi_allan_umass_edu/rsenapati/run_experiments.py --prompt_type BASIC --attack_type none --mitigation_type none --output_dir data --limit 10

