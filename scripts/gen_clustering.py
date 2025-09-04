for embedding in ["clip","ssl","siglip2","dino"]:
    command=f" sbatch runpymain.sh -J clust --err=slurm/clustering/{embedding}.err --out=slurm/clustering/{embedding}.out "
    command+=f" clustering.py --src_dataset jlbaker361/league-tagged-{embedding} --dest_dataset jlbaker361/league-{embedding}-clustering "
    print(command)