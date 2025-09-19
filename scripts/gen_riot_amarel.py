for embedding in ["clip","ssl","siglip2","dino"]:
    command=f"sbatch -J riot --err=slurm/riot/{embedding}.err --out=slurm/riot/{embedding}.out runpygpu.sh riot_embedding.py --embedding {embedding} --dest_dataset jlbaker361/league-tagged-{embedding} "
    print(command)