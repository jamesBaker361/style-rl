for training_type in ["denoise","reward"]:
    for prediction_type in ["epsilon","v_prediction"]:
        for embedding in ["clip"]:
            if prediction_type=="epsilon" and training_type=="reward":
                continue
            for pipeline in ["sana","lcm"]:
                for suffix in ["none","vanilla"]:
                    name=f"{training_type}_{prediction_type}_{embedding}_{pipeline}_{suffix}"

                    command=f"sbatch  -J pers --exclude=gpu[005,006,010-014,017,018],cuda[001-008],pascal[001-010],gpuk[001-012] --err=slurm/test/{name}.err --out=slurm/test/{name}.out"
                    command+=" runpygpu.sh main_pers.py --epochs 3 --limit 10 --project_name testing-pers --reward_switch_epoch 2 "
                    command+=f" --mixed_precision fp16 --prediction_type {prediction_type} --pipeline {pipeline} --num_inference_steps 2 "
                    command+=f" --embedding {embedding} --training_type {training_type} --dataset jlbaker361/{embedding}-art_coco_captioned-50"
                    if pipeline=="sana":
                        command+=f"-{pipeline} "
                    if suffix=="vanilla":
                        command+=" --vanilla "
                    print(command)