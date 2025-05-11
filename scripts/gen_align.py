for start in [0]:
    limit=start+4
    for version in ["content","style3","baseline" ,"style3_stable"]: #"content","baseline","both"]:
        flags={
            "style3":" --style_layers_train --style_layers 3 1 2 ",
            "style2":" --style_layers_train --style_layers 0 1 2 ",
            "style":" --style_layers_train --style_layers 0 1 ",
            "content":" --content_layers_train ",
            "both":" --style_layers_train --style_layers 1 2 3  --content_layers_train ",
            "baseline":"  --style_layers_train --style_layers 0 1 2 3  --style_mid_block ",
            "style3_stable":" --style_layers_train --style_layers 3 1 2  --pretrained_type stable --num_inference_steps 50 ",
            "one_shot":"--style_layers_train --style_layers 3 2 --style_mid_block"
        }[version]
        for lr in [0.001]:
            if version=="style2" and lr==0.001:
                continue
            for reward in ["vgg"]:
                for method in ["align"]:
                    for epochs in [25]:
                        if method=="hook" and reward=="vgg":
                            continue
                        
                        method_flags={
                            "ddpo":" --mixed_precision no --gradient_accumulation_steps 2 ",
                            "hook":" --mixed_precision bf16 --gradient_accumulation_steps 16 ",
                            "align":" --mixed_precision bf16 --gradient_accumulation_steps 8 "
                        }[method]
                        if version=="both":
                            epochs=25
                        name=f"{reward}_{version}_{start}_{lr}_{method}_{epochs}"
                        exclude=" --exclude=gpu[005,006,010-014,017,018],cuda[001-008],pascal[001-010] " #gpu[011-014,017-018,021,027-028,005-010],cuda[001-008]
                        command=f" sbatch -J align {exclude} --err=slurm/aligntest768/{name}.err --out=slurm/aligntest768/{name}.out runpygpu.sh main.py --batch_size 1 --project_name style-rl "
                        command+=f" --start {start} --limit {limit} {flags}  --sample_num_batches_per_epoch 16 --image_size 768 --epochs {epochs}  --method {method} "
                        command+=f" --reward_fn {reward}  "
                        command+=f" --learning_rate {lr} {method_flags} "
                        print(command)