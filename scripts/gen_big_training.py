port=29601

for training_type in  ["denoise","reward"]:
    for prediction_type in ["epsilon"]:
        for embedding in ["dino","ssl","clip","siglip2"]:
            for data in ["league_captioned_tile"]:
                name=f"{training_type}_{prediction_type}_{embedding}"
                port+=1
                command=f"sbatch  -J pers  --err=slurm/pers_{data}/{name}.err --out=slurm/pers_{data}/{name}.out"
                command+=f" runaccgpu.sh --multi_gpu --mixed_precision fp16 --num_processes 2 --main_process_port {port} main_pers.py --epochs 100 --limit -1 --batch_size 4 --project_name {data}-20 "
                command+=f" --mixed_precision fp16 --prediction_type {prediction_type} --upload_interval 10 "
                command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-{data}-20 --vanilla --name jlbaker361/{name} --gradient_accumulation_steps 8 "
                print(command)