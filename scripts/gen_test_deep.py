port=29501

for training_type in ["denoise","reward"]:
    for prediction_type in ["epsilon","v_prediction"]:
        for embedding in ["dino","ssl","siglip2","clip"]:
            if prediction_type=="epsilon" and training_type=="reward":
                continue
            name=f"{training_type}_{prediction_type}_{embedding}"
            port+=1
            command=f"sbatch  -J pers  --err=slurm/test_deep/{name}.err --out=slurm/test_deep/{name}.out"
            command+=f" runaccgpu.sh --multi_gpu --mixed_precision fp16 --num_processes 2 --main_process_port {port} main_pers.py --epochs 4 --limit 100 --batch_size 4 --project_name testing-pers "
            command+=f" --mixed_precision fp16 --prediction_type {prediction_type} "
            command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-art_coco_captioned --vanilla --name jlbaker361/{name} --gradient_accumulation_steps 4 "
            print(command)