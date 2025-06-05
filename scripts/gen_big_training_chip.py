port=29601

for training_type in  ["denoise"]:
    for prediction_type in ["epsilon"]:
        for embedding in ["clip"]:
            for data in ["art_coco_captioned"]:
                for frac in [1]:
                    for lr in [0.001,0.0001]:
                        n=500
                        name=f"{training_type}_{prediction_type}_{embedding}_{frac}_{lr}_{n}"
                        port+=1
                        command=f"sbatch  --gres=gpu:1  -J pers  --err=slurm_chip/pers_{data}/{name}.err --out=slurm_chip/pers_{data}/{name}.out"
                        command+=f" runaccgpu_chip.sh  --mixed_precision fp16 --num_processes 1 --main_process_port {port} main_pers.py --epochs 5000 --limit -1 --batch_size 2 --project_name {data}-{n} "
                        command+=f" --mixed_precision fp16 --prediction_type {prediction_type} --upload_interval 100 --uncaptioned_frac {frac} --train_split 0.8 --lr {lr} --load  --generic_test_prompts "
                        command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-{data}-{n} --vanilla --name jlbaker361/{name} --gradient_accumulation_steps 8 "
                        print(command)