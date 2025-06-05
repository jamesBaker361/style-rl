port=29650

for training_type in  ["denoise"]: #,"reward","latents_reward"]:
    for frac in [1.0]:
        for prediction_type in ["epsilon"]:
            for embedding in ["clip"]:
                for data in ["league_captioned_splash"]:
                    for lr in [0.001,0.0001]:
                        for n in [500,1000]:
                            for suffix in ["_no_proj","_identity"]:
                                name=f"{training_type}_{prediction_type}_{embedding}_{frac}_{lr}_{n}{suffix}"
                                port+=1
                                command=f"sbatch  -J pers  --err=slurm/pers_{data}/{name}.err --out=slurm/pers_{data}/{name}.out"
                                command+=f" runaccgpu.sh --multi_gpu --mixed_precision fp16 --num_processes 2 --main_process_port {port} main_pers.py --epochs 400 --limit -1 --batch_size 2 --project_name {data}-{n} "
                                command+=f" --mixed_precision fp16 --prediction_type {prediction_type} --upload_interval 100 --uncaptioned_frac {frac} --train_split 0.95 --lr {lr} --load --generic_test_prompts "
                                command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-{data}-{n} --vanilla --name jlbaker361/{name} --gradient_accumulation_steps 4  "
                                if suffix=="_no_proj":
                                    command+=" --disable_projection_adapter "
                                elif suffix=="_identity":
                                    command+=" --identity_adapter "
                                print(command)