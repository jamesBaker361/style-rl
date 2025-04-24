
sbatch -J pers --err=slurm/main/pers-ssl.err --out=slurm/main/pers-ssl.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --dataset jlbaker361/captioned-test
sbatch -J pers --err=slurm/main/pers-ssl-velocity.err --out=slurm/main/pers-ssl-velocity.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --prediction_type velocity --dataset jlbaker361/captioned-test

sbatch -J pers --err=slurm/main/pers-ssl-reward.err --out=slurm/main/pers-ssl-reward.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --training_type reward --dataset jlbaker361/captioned-test