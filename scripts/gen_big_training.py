port=29650

for training_type in  ["mse_reward","denoise"]: #,"reward","latents_reward"]:
    for frac in [1.0,0.0]:
        for prediction_type in ["epsilon"]:
            for embedding in ["clip"]:
                for data in ["league_captioned_splash"]:
                    name=f"{training_type}_{prediction_type}_{embedding}_{frac}"
                    port+=1
                    command=f"sbatch  -J pers  --err=slurm/pers_{data}/{name}.err --out=slurm/pers_{data}/{name}.out"
                    command+=f" runaccgpu.sh --multi_gpu --mixed_precision fp16 --num_processes 2 --main_process_port {port} main_pers.py --epochs 5000 --limit -1 --batch_size 2 --project_name {data}-50 "
                    command+=f" --mixed_precision fp16 --prediction_type {prediction_type} --upload_interval 100 --uncaptioned_frac {frac} --train_split 0.8 "
                    command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-{data}-50 --vanilla --name jlbaker361/{name} --gradient_accumulation_steps 4 --load "
                    print(command)