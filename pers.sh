
sbatch -J pers --err=slurm/main/pers-ssl.err --out=slurm/main/pers-ssl.out runpymain.sh main_pers.py --epochs 2 --embedding ssl

sbatch -J pers --err=slurm/main/pers-ssl.err --out=slurm/main/pers-ssl.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --training_type reward