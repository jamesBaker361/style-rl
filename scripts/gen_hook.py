for start in [0]:
    limit=start+4
    for version in ["style2" ,"style"]: #"content","baseline","both"]:
        flags={
            "style2":" --style_layers_train --style_layers 0 1 2 ",
            "style":" --style_layers_train --style_layers 0 1 ",
            "content":" --content_layers_train ",
            "both":" --style_layers_train --style_layers 0 1  --content_layers_train ",
            "baseline":"  --style_layers_train --style_layers 0 1 2 3  --style_mid_block "
        }[version]
        for lr in [0.001,0.01]:
            for reward in ["mse","vgg"]:
                for method in ["hook","ddpo","align"]:
                    if method=="hook" and reward=="vgg":
                        continue
                    name=f"{reward}_{version}_{start}_{lr}_{method}"
                    method_flags={
                        "ddpo":" --mixed_precision no --gradient_accumulation_steps 2 ",
                        "hook":" --mixed_precision bf16 --gradient_accumulation_steps 16 ",
                        "align":" --mixed_precision bf16 --gradient_accumulation_steps 16 "
                    }[method]
                    exclude=" --exclude=gpu[005,006,010-014,017,018],cuda[001-008],pascal[001-010] " #gpu[011-014,017-018,021,027-028,005-010],cuda[001-008]
                    command=f" sbatch -J style {exclude} --err=slurm/style/{name}.err --out=slurm/style/{name}.out runpygpu.sh main.py --batch_size 1  "
                    command+=f" --start {start} --limit {limit} {flags}  --sample_num_batches_per_epoch 16 --image_size 256 --epochs 50  --method {method} "
                    command+=f" --reward_fn {reward}  "
                    command+=f" --learning_rate {lr} {method_flags} "
                    print(command)