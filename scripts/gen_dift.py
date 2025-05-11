for content_start in [0]:
    content_limit=content_start+2
    for version in ["content_all"]:
        flags={
            "content_mid": " --content_layers_train --content_mid_block ",
            "content_mid_3": " --content_layers_train --content_mid_block --content_layers 3  ", 
            "style_content": "--content_layers_train --content_mid_block --content_layers 3 --style_layers_train --style_layers 0 1",
            "content_all": " --content_layers_train --content_mid_block --content_layers 3 2 1 0  "
        }[version]

        for lr in [0.0001, 0.001]:
            for content_reward_fn in ["dift"]:
                for method in ["align"]:
                    for epochs in [500]:
                        for embedding in ["normal"]:
                            method_flags={
                            "ddpo":" --mixed_precision no --gradient_accumulation_steps 2 ",
                            "hook":" --mixed_precision bf16 --gradient_accumulation_steps 16 ",
                            "align":" --mixed_precision bf16 --gradient_accumulation_steps 12 "
                            }[method]
                            embedding_flag={
                                "prompt":"--prompt_embedding_conditioning",
                                "encoder":"--use_encoder_hid_proj",
                                "normal":""
                            }[embedding]
                            for up_ft_index in [0,1,2]:
                                for t in [261,500]:

                                    name=f"{version}_{lr}_{content_reward_fn}_{embedding}_{epochs}_{up_ft_index}_{t}"
                                    exclude=" --exclude=gpu[005,006,008,010-014,017,018],cuda[001-008],pascal[001-010] " 
                                    command=f" sbatch -J align {exclude} --err=slurm/aligndift/{name}.err --out=slurm/aligndift/{name}.out runpygpu.sh main.py --batch_size 1 --prompt portrait --project_name content-rl-test "
                                    command+=f" --content_start {content_start} --content_limit {content_limit} {flags}  --sample_num_batches_per_epoch 256 --image_size 256 --epochs {epochs}  --method {method} --limit 1"
                                    command+=f" --content_reward_fn {content_reward_fn}  "
                                    command+=f" --learning_rate {lr} {method_flags} {embedding_flag}   "
                                    command+=f"--up_ft_index {up_ft_index} --t {t}"
                                    print(command)