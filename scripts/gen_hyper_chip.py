for embedding in ["clip","ssl","siglip2","dino"]:
    command=f"sbatch -J hyper --err=slurm_chip/{embedding}.err --out=slurm_chip/{embedding}.out runpygpu_chip.sh hyper_plane.py "
    command+=f" --src_dataset jlbaker361/league-tagged-{embedding} --dest_dataset jlbaker361/league-{embedding}-classification"