for lr in [0.001,0.0001]:
    for data in ["league_captioned_splash"]:
        for n in [1000]:
            for retain_fraction in [0.25,0.5,1.0]:
                name=f"scale_{lr}_{data}_{n}_{retain_fraction}"
                command=f"sbatch -J scale --err=slurm/scale/{name}.err --out=slurm/scale/{name}.out runpygpu.sh  scale_prediction.py "
                command+=f" --epochs 100  --validation_interval 5 --identity_adapter --train_split 0.95 --limit -1"
                command+=f" --batch_size 2 --gradient_accumulation_steps 16 --project_name {data}-{n}-scale "
                command+=f" --dataset jlbaker361/clip-{data}-{n} --lr {lr}   --use_lora --load --name {name} --retain_fraction {retain_fraction} "
                print(command)