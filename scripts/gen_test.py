for training_type in ["denoise","reward"]:
    for prediction_type in ["epsilon","v_prediction"]:
        for embedding in ["dino","ssl","siglip2","clip"]:
            if prediction_type=="epsilon" and training_type=="reward":
                continue
            name=f"{training_type}_{prediction_type}_{embedding}"
            command=f"sbatch  -J pers --exclude=gpu[005,006,010-014,017,018],cuda[001-008],pascal[001-010] --err=slurm/test/{name}.err --out=slurm/test/{name}.out"
            command+=" runpygpu.sh main_pers.py --epochs 2 --limit 10 --project_name testing-pers "
            command+=f" --mixed_precision fp16 --prediction_type {prediction_type} "
            command+=f" --embedding {embedding} --training_type {training_type} "
            print(command)