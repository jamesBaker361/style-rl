port=29650

for training_type in  ["reward"]: #,"reward","latents_reward"]:
    for frac in [1.0]:
        for prediction_type in ["v_prediction"]:
            for embedding in ["clip",]:
                for data in ["art_coco_captioned"]:
                    for suffix in ["_deep_identity",""]:
                        for lr in [0.001,0.0001]:
                            for n in [1000]:
                                #for scheduler in ["LCMScheduler"]:
                                for pipeline in ["sana"]:
                        
                                    name=f"{training_type}_{prediction_type}_{embedding}_{frac}_{lr}_{n}{suffix}_{pipeline}"
                                    port+=1
                                    command=f"sbatch  -J pers  --err=slurm_chip/pers_{data}/{name}.err --out=slurm_chip/pers_{data}/{name}.out --gres=gpu:1 "
                                    command+=f" runaccgpu_chip.sh --mixed_precision fp16 --num_processes 1 --main_process_port {port} main_pers.py --epochs 500 --limit -1 --batch_size 2 --project_name {data}-{n} "
                                    command+=f" --mixed_precision fp16 --prediction_type {prediction_type} --upload_interval 5 --uncaptioned_frac {frac} --train_split 0.95 --lr {lr} --load --generic_test_prompts "
                                    command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-{data}-{n}-{pipeline} --vanilla --name jlbaker361/{name} --gradient_accumulation_steps 8  "
                                    command+=f" --num_inference_steps 2 "
                                    if training_type=="reward":
                                        command+=" --initial_scale 0.25 "
                                    if suffix=="_no_proj":
                                        command+=" --disable_projection_adapter "
                                    elif suffix=="_identity":
                                        command+=" --identity_adapter "
                                    elif suffix=="_deep_identity":
                                        command+=" --identity_adapter  --deep_to_ip_layers "
                                    print(command)