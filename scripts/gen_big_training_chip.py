port=29601

for training_type in  ["denoise"]:
    for prediction_type in ["epsilon"]:
        for embedding in ["dino","clip",]:
            for data in ["art_coco_captioned"]:
                if prediction_type=="epsilon" and training_type=="reward":
                    continue
                name=f"{training_type}_{prediction_type}_{embedding}"
                port+=1
                command=f"sbatch  -J pers  --err=slurm_chip/pers_{data}/{name}.err --out=slurm_chip/pers_{data}/{name}.out"
                command+=f" runaccgpu_chip.sh --multi_gpu --mixed_precision fp16 --num_processes 2 --main_process_port {port} main_pers.py --epochs 250 --limit -1 --batch_size 4 --project_name {data} "
                command+=f" --mixed_precision fp16 --prediction_type {prediction_type} "
                command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-art_coco_captioned --vanilla --name jlbaker361/{name} --gradient_accumulation_steps 8 "
                print(command)