
sbatch -J pers --err=slurm/main/pers-ssl.err --out=slurm/main/pers-ssl.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --limit 8
sbatch -J pers --err=slurm/main/pers-ssl-velocity.err --out=slurm/main/pers-ssl-velocity.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --prediction_type velocity --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --limit 8

sbatch -J pers --err=slurm/main/pers-ssl-reward.err --out=slurm/main/pers-ssl-reward.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --training_type reward --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --buffer_size 0 --limit 8


sbatch -J pers --err=slurm/main/gpu-pers-ssl.err --out=slurm/main/gpu-pers-ssl.out runpygpu.sh main_pers.py --epochs 2 --embedding ssl --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --mixed_precision bf16 --limit 8
sbatch -J pers --err=slurm/main/gpu-pers-ssl-velocity.err --out=slurm/main/gpu-pers-ssl-velocity.out runpygpu.sh main_pers.py --epochs 2 --embedding ssl --prediction_type velocity --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --mixed_precision bf16 --limit 8

sbatch -J pers --err=slurm/main/gpu-pers-ssl-reward.err --out=slurm/main/gpu-pers-ssl-reward.out runpygpu.sh main_pers.py --epochs 2 --embedding ssl --training_type reward --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --buffer_size 0 --mixed_precision bf16 --limit 8