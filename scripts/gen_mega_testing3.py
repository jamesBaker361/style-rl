port=29650

for training_type in  ["denoise"]: #,"reward","latents_reward"]:
    for frac in [1.0]:
        for prediction_type in ["epsilon"]:
            for embedding in ["clip","dino","ssl","siglip2"]:
                for data in ["league_captioned_splash"]:
                    for suffix in ["_identity"]:
                        for lr in [0.001]:
                            for n in [1000]:
                                #for scheduler in ["LCMScheduler"]:
                                for pipeline in ["lcm"]:
                                    for reward_switch_epoch in [-1]:
                        
                                        name=f"{training_type}_{prediction_type}_{embedding}_{frac}_{lr}_{n}{suffix}_{pipeline}_{reward_switch_epoch}"
                                        port+=1
                                        for [start,end] in [[0.25,0.75], [0.5,1.0], [0.0,0.5]]:
                                            command=f"sbatch  -J perstest  --err=slurm/perstesting_{data}/{start}_{name}.err --out=slurm/perstesting_{data}/{start}_{name}.out --gres=gpu:1 "
                                            command+=f" runaccgpu.sh  --mixed_precision fp16 --num_processes 1 --main_process_port {port} main_logging.py --epochs 500 --limit -1 --batch_size 2 --project_name {data}-{n}-testing-start "
                                            command+=f" --mixed_precision fp16 --prediction_type {prediction_type} --upload_interval 1 --uncaptioned_frac {frac} --train_split 0.95 --lr {lr} --load --generic_test_prompts "
                                            command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-{data}-{n} --vanilla --name jlbaker361/{name} --gradient_accumulation_steps 8  "
                                            command+=f" --pipeline {pipeline} --reward_switch_epoch {reward_switch_epoch}   --real_test_prompts  --num_inference_steps 8 "
                                            command+=f" --ip_start {start} --ip_end {end} "
                                            if training_type=="reward":
                                                command+=" --initial_scale 0.25 "
                                            if suffix=="_no_proj":
                                                command+=" --disable_projection_adapter "
                                            elif suffix=="_identity":
                                                command+=" --identity_adapter "
                                            elif suffix=="_deep_identity":
                                                command+=" --identity_adapter  --deep_to_ip_layers "
                                            print(command)