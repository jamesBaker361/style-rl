for content_start in [0]:
    content_limit=content_start+2
    for version in ["content_mid", "content_mid_3","content_all"]:
        flags={
            "content_mid": " --content_layers_train --content_mid_block ",
            "content_mid_3": " --content_layers_train --content_mid_block --content_layers 3  ", 
            "style_content": "--content_layers_train --content_mid_block --content_layers 3 --style_layers_train --style_layers 0 1",
            "content_all": " --content_layers_train --content_mid_block --content_layers 3 2 1 0  "
        }[version]

        for lr in [0.00001, 0.00005]:
            for content_reward_fn in ["mse","face"]:
                for method in ["align"]:
                    for epochs in [250]:
                        if method=="hook" and content_reward_fn=="vgg":
                            continue
                        
                        method_flags={
                            "ddpo":" --mixed_precision no --gradient_accumulation_steps 2 ",
                            "hook":" --mixed_precision bf16 --gradient_accumulation_steps 16 ",
                            "align":" --mixed_precision bf16 --gradient_accumulation_steps 12 "
                        }[method]
                        name=f"{content_reward_fn}_{version}_{content_start}_{lr}_{method}_{epochs}"
                        exclude=" --exclude=gpu[005,006,008,010-014,017,018],cuda[001-008],pascal[001-010] " #gpu[011-014,017-018,021,027-028,005-010],cuda[001-008]
                        command=f" sbatch -J align {exclude} --err=slurm/aligntest768/{name}.err --out=slurm/aligntest768/{name}.out runpygpu.sh main.py --batch_size 1 --prompt portrait --project_name content-rl-test "
                        command+=f" --content_start {content_start} --content_limit {content_limit} {flags}  --sample_num_batches_per_epoch 256 --image_size 768 --epochs {epochs}  --method {method} --limit 1"
                        command+=f" --content_reward_fn {content_reward_fn}  "
                        command+=f" --learning_rate {lr} {method_flags} "
                        print(command)