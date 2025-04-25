
sbatch -J perstest --err=slurm/main/pers-siglip2.err --out=slurm/main/pers-siglip2.out runpymain.sh main_pers.py --epochs 2 --embedding siglip2 --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --limit 8 --gradient_accumulation_steps 2
sbatch -J perstest --err=slurm/main/pers-siglip2-v_prediction.err --out=slurm/main/pers-siglip2-v_prediction.out runpymain.sh main_pers.py --epochs 2 --embedding siglip2 --prediction_type v_prediction --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --limit 8 --gradient_accumulation_steps 2

sbatch -J perstest --err=slurm/main/pers-siglip2-reward.err --out=slurm/main/pers-siglip2-reward.out runpymain.sh main_pers.py --epochs 2 --embedding siglip2 --training_type reward --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --buffer_size 0 --limit 8 --gradient_accumulation_steps 2


sbatch -J perstest --err=slurm/main/gpu-pers-siglip2.err --out=slurm/main/gpu-pers-siglip2.out runpygpu.sh main_pers.py --epochs 2 --embedding siglip2 --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --mixed_precision fp16 --limit 8 --gradient_accumulation_steps 2
sbatch -J perstest --err=slurm/main/gpu-pers-siglip2-v_prediction.err --out=slurm/main/gpu-pers-siglip2-v_prediction.out runpygpu.sh main_pers.py --epochs 2 --embedding siglip2 --prediction_type v_prediction --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --mixed_precision fp16 --limit 8 --gradient_accumulation_steps 2

sbatch -J perstest --err=slurm/main/gpu-pers-siglip2-reward.err --out=slurm/main/gpu-pers-siglip2-reward.out runpygpu.sh main_pers.py --epochs 2 --embedding siglip2 --training_type reward --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --buffer_size 0 --mixed_precision fp16 --limit 8 --gradient_accumulation_steps 2


sbatch -J perstest --err=slurm/main/pers-ssl.err --out=slurm/main/pers-ssl.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --limit 8 --gradient_accumulation_steps 2
sbatch -J perstest --err=slurm/main/pers-ssl-v_prediction.err --out=slurm/main/pers-ssl-v_prediction.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --prediction_type v_prediction --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --limit 8 --gradient_accumulation_steps 2

sbatch -J perstest --err=slurm/main/pers-ssl-reward.err --out=slurm/main/pers-ssl-reward.out runpymain.sh main_pers.py --epochs 2 --embedding ssl --training_type reward --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --buffer_size 0 --limit 8 --gradient_accumulation_steps 2


sbatch -J perstest --err=slurm/main/gpu-pers-ssl.err --out=slurm/main/gpu-pers-ssl.out runpygpu.sh main_pers.py --epochs 2 --embedding ssl --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --mixed_precision bf16 --limit 8 --gradient_accumulation_steps 2
sbatch -J perstest --err=slurm/main/gpu-pers-ssl-v_prediction.err --out=slurm/main/gpu-pers-ssl-v_prediction.out runpygpu.sh main_pers.py --epochs 2 --embedding ssl --prediction_type v_prediction --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --mixed_precision bf16 --limit 8 --gradient_accumulation_steps 2

sbatch -J perstest --err=slurm/main/gpu-pers-ssl-reward.err --out=slurm/main/gpu-pers-ssl-reward.out runpygpu.sh main_pers.py --epochs 2 --embedding ssl --training_type reward --dataset jlbaker361/captioned-test --validation_interval 1 --train_split 0.5 --image_size 128 --buffer_size 0 --mixed_precision bf16 --limit 8 --gradient_accumulation_steps 2