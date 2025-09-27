import os
from pathlib import Path

# Experiment choices
PROMPT_TYPES = ["BASIC", "RATIONALE", "UTILITY"]
ATTACK_TYPES = ["none", "prepend", "append", "scatter"]
MITIGATION_TYPES = ["none", "user_prompt_hardening", "system_prompt_hardening", "few_shot"]


BASELINE_TEMPLATE = """#!/bin/bash

#SBATCH -p gpu  # Submit job to the gpu partition
#SBATCH -t 48:00:00     # Set maximum job time to 2 days 
#SBATCH --mail-type=TIME_LIMIT_80 # Email me if time exceed 80 percent of the allocated time or 38.4 hours
#SBATCH --gpus=1        # Request 1 GPU
#SBATCH --output=gpu_job_{prompt}_{attack}_{mitigation}.out  # Name the output file with the job ID and experiment choices

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
python /work/pi_allan_umass_edu/rsenapati/run_experiments.py --prompt_type {prompt} --attack_type {attack} --mitigation_type {mitigation} --output_dir /work/pi_allan_umass_edu/rsenapati/data

"""


def main() -> None:
    project_root = Path(__file__).parent
    run_scripts_dir = project_root / "run_scripts"
    run_scripts_dir.mkdir(parents=True, exist_ok=True)

    num_created = 0
    for prompt in PROMPT_TYPES:
        for attack in ATTACK_TYPES:
            for mitigation in MITIGATION_TYPES:
                """# skip baseline script
                if prompt == "BASIC" and attack == "none" and mitigation == "none":
                    continue
                """
                script_name = f"run_{prompt}_{attack}_{mitigation}.sh"
                script_path = run_scripts_dir / script_name
                content = BASELINE_TEMPLATE.format(prompt=prompt, attack=attack, mitigation=mitigation)
                # Force LF line endings so SLURM does not complain about DOS line breaks
                with open(script_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(content)
                try:
                    os.chmod(script_path, 0o755)
                except Exception:
                    pass
                num_created += 1

    print(f"Created {num_created} run scripts in: {run_scripts_dir}")


if __name__ == "__main__":
    main()
