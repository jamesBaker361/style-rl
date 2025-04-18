
#sbatch --err=slurm/main/both_cpu.err --out=slurm/main/both_cpu.out runpymain.sh main.py --style_layers_train --content_layers_train --epochs 1   --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --batch_size 1 

#sbatch --err=slurm/main/style_cpu.err --out=slurm/main/style_cpu.out runpymain.sh main.py --style_layers_train --epochs 1   --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --batch_size 1 --project_name style_test
#sbatch --exclude=gpu[005,006,011,012],cuda[001,002] --err=slurm/main/style_gpu.err --out=slurm/main/style_gpu.out runpygpu.sh main.py --style_layers_train --epochs 1 --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --mixed_precision bf16 --batch_size 1 --project_name style_test
#sbatch --exclude=gpu[005,006,011,012],cuda[001,002] --err=slurm/main/both_gpu.err --out=slurm/main/both_gpu.out runpygpu.sh main.py --style_layers_train --content_layers_train --epochs 1 --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --mixed_precision bf16  --batch_size 1

#sbatch --exclude=gpu[005,006,011,012],cuda[001,002] --err=slurm/main/eval_gpu.err --out=slurm/main/eval_gpu.out runpygpu.sh main.py --style_layers_train --epochs 0 --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --mixed_precision fp16 --batch_size 1

#sbatch --exclude=gpu[005,006,011-013,017],cuda[001-008],pascal[001-010] --err=slurm/main/align_style_gpu.err --out=slurm/main/align_style_gpu.out runpygpu.sh main.py --style_layers_train --epochs 1 --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --mixed_precision bf16 --batch_size 1 --method align
#sbatch --err=slurm/main/hook_cpu_mse.err --out=slurm/main/hook_cpu_mse.out runpymain.sh main.py --style_layers_train --epochs 2   --gradient_accumulation_steps 4 --sample_num_batches_per_epoch 2 --batch_size 1 --method hook --num_inference_steps 2 --image_size 128 --reward_fn mse --project_name style_test


#sbatch --err=slurm/main/style_cpu_mse.err --out=slurm/main/style_cpu_mse.out runpymain.sh main.py --style_layers_train --epochs 1   --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --batch_size 1 --reward_fn mse --project_name style_test
#sbatch  --err=slurm/main/align_style_cpu_mse.err --out=slurm/main/align_style_cpu_mse.out runpymain.sh main.py --style_layers_train --epochs 1 --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2  --batch_size 1 --method align --reward_fn mse --project_name style_test

#sbatch --err=slurm/main/style_cpu_vgg.err --out=slurm/main/style_cpu_vgg.out runpymain.sh main.py --style_layers_train --epochs 1   --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --batch_size 1 --reward_fn vgg --project_name style_test
#sbatch  --err=slurm/main/align_style_cpu_vgg.err --out=slurm/main/align_style_cpu_vgg.out runpymain.sh main.py --style_layers_train --epochs 1 --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2  --batch_size 1 --method align --reward_fn vgg --project_name style_test

sbatch --err=slurm/main/ir_cpu.err --out=slurm/main/ir_cpu.out runpymain.sh main_ir.py --style_layers_train --epochs 1   --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --batch_size 1 --project_name style_test --reward_fn ir --style_layers_train --style_mid_block --prompt_src_txt random_animals.txt
sbatch --exclude=gpu[005,006,011,012],cuda[001,002] --err=slurm/main/ir_gpu.err --out=slurm/main/ir_gpu.out runpygpu.sh main_ir.py --style_layers_train --epochs 1 --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --mixed_precision bf16 --batch_size 1 --project_name style_test --reward_fn ir --style_layers_train --style_mid_block --prompt_src_txt random_animals.txt
sbatch --err=slurm/main/ir_cpu_inv.err --out=slurm/main/ir_cpu_inv.out runpymain.sh main_ir.py --style_layers_train --epochs 1   --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --batch_size 1 --project_name style_test --reward_fn ir --style_layers_train --style_mid_block --prompt_src_txt random_animals.txt --textual_inversion
sbatch --exclude=gpu[005,006,011,012],cuda[001,002] --err=slurm/main/ir_gpu_inv.err --out=slurm/main/ir_gpu_inv.out runpygpu.sh main_ir.py --style_layers_train --epochs 1 --gradient_accumulation_steps 1 --sample_num_batches_per_epoch 2 --mixed_precision bf16 --batch_size 1 --project_name style_test --reward_fn ir --style_layers_train --style_mid_block --prompt_src_txt random_animals.txt --textual_inversion
