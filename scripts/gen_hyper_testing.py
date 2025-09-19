port=29650

for training_type in  ["denoise"]: #,"reward","latents_reward"]:
    for frac in [1.0]:
        for prediction_type in ["epsilon"]:
            for embedding in ["clip"]:
                for data in ["league_captioned_splash"]:
                    for suffix in ["_identity"]:
                        for lr in [0.001]:
                            for n in [1000]:
                                #for scheduler in ["LCMScheduler"]:
                                for pipeline in ["lcm"]:
                                    for reward_switch_epoch in [-1]:
                                        for hyperplane_coefficient in [-10.0,-2,2,10]:
                                            for classifier_type in ["SGD","SVC"]:
                        
                                                name=f"{training_type}_{prediction_type}_{embedding}_{frac}_{lr}_{n}{suffix}_{pipeline}_{reward_switch_epoch}"
                                                port+=1
                                                command=f"sbatch  -J perstest  --err=slurm/perstesting_{data}_hyper/{hyperplane_coefficient}_{classifier_type}_{name}.err --out=slurm/perstesting_{data}_hyper/{hyperplane_coefficient}_{classifier_type}_{name}.out --gres=gpu:1 "
                                                command+=f" runaccgpu.sh  --mixed_precision fp16 --num_processes 1 --main_process_port {port} main_logging.py  --limit -1 --batch_size 2 --project_name hyper "
                                                command+=f" --mixed_precision fp16   --uncaptioned_frac {frac} --train_split 0.5  --generic_test_prompts "
                                                command+=f" --embedding {embedding}  --dataset jlbaker361/{embedding}-{data}-{n} --vanilla --name jlbaker361/{name}  "
                                                command+=f" --pipeline {pipeline}   --num_inference_steps 8 --hyperplane --hyperplane_coefficient {hyperplane_coefficient} "
                                                command+=f" --classifier_type {classifier_type} "
                                                if suffix=="_no_proj":
                                                    command+=" --disable_projection_adapter "
                                                elif suffix=="_identity":
                                                    command+=" --identity_adapter "
                                                elif suffix=="_deep_identity":
                                                    command+=" --identity_adapter  --deep_to_ip_layers "
                                                print(command)