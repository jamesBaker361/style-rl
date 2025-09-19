for embedding in ["clip","ssl","siglip2","dino"]:
    command=f" sbatch -J ipa --err=slurm/ipattn/{embedding}.err --out=slurm/ipattn/{embedding}.out runpygpu.sh ipattn.py {embedding} "
    print(command)