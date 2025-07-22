for lr in [0.001,0.0001]:
    for embedding in ["clip","ssl"]:
        for data in ["league_captioned_splash"]:
            for n in [1000]:
                name=f"{embedding}_scale_{lr}_{data}_{n}"
                command=f"sbatch -J scale --err=slurm_chip/scale/{name}.err --out=slurm_chip/scale/{name}.out runpygpu_chip.sh  scale_prediction.py "
                command+=f" --epochs 100  --validation_interval 5 --identity_adapter --train_split 0.95 --limit -1"
                command+=f" --batch_size 2 --gradient_accumulation_steps 16 --project_name {data}-{n}-scale "
                command+=f" --dataset jlbaker361/{embedding}-{data}-{n} --lr {lr} --embedding {embedding}  --use_lora --load --name {name} "
                print(command)