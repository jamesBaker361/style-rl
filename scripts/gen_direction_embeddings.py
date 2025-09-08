for seed_offset in [x *10 for x in range(10)]:
    for embedding in ["clip","dino","ssl","siglip2"]:
        command=f"sbatch -J dir --err=slurm_chip/direction_emb/{embedding}_{seed_offset}.err --out=slurm_chip/direction_emb/{embedding}_{seed_offset}.out "
        command+=f" runpygpu_chip.sh directional_embeddings.py  --src_dataset jlbaker361/directional-{seed_offset} "
        command+=f" --upload_interval 50 "
        command+=f" --embedding {embedding} --dest_dataset jlbaker361/directional-{seed_offset}-{embedding} "
        print(command)