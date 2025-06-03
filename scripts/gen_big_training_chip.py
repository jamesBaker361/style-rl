port=29601

for training_type in  ["denoise","mse_reward"]:
    for prediction_type in ["epsilon"]:
        for embedding in ["clip"]:
            for data in ["art_coco_captioned"]:
                for frac in [1,0]:
                    name=f"{training_type}_{prediction_type}_{embedding}_{frac}"
                    port+=1
                    command=f"sbatch  --gres=gpu:1  -J pers  --err=slurm_chip/pers_{data}/{name}.err --out=slurm_chip/pers_{data}/{name}.out"
                    command+=f" runaccgpu_chip.sh  --mixed_precision fp16 --num_processes 1 --main_process_port {port} main_pers.py --epochs 5000 --limit -1 --batch_size 2 --project_name {data}-50 "
                    command+=f" --mixed_precision fp16 --prediction_type {prediction_type} --upload_interval 100 --uncaptioned_frac {frac} --train_split 0.8 "
                    command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-{data}-50 --vanilla --name jlbaker361/{name} --gradient_accumulation_steps 8 "
                    print(command)