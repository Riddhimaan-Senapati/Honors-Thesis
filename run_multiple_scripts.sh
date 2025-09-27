
for script in /work/pi_allan_umass_edu/rsenapati/run_scripts/*.sh; do     sbatch "$script" -q long; done