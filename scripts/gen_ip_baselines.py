for scale in [0.5,1.0]:
    for data in ["mtg","dreambooth"]:
        name=f"ip_{scale}_{data}"
        command=f"sbatch -J ip --out=slurm/eval/{name}.out --err=slurm/eval/{name}.err runpygpu.sh main_ip.py --scale {scale} --src_dataset jlbaker361/{data} --dest_dataset jlbaker361/{name}  "
        print(command)