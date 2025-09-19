for embedding in ["clip","ssl","siglip2","dino"]:
    command=f"sbatch -J hyper --err=slurm/hyper/{embedding}.err --out=slurm/hyper/{embedding}.out runpygpu.sh hyper_plane.py "
    command+=f" --src_dataset jlbaker361/league-tagged-{embedding} --dest_dataset jlbaker361/league-{embedding}-classification"
    print(command)