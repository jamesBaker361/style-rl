for embedding in ["siglip2","ssl","clip","dino"]:
    for data in ["league_captioned_splash","coco_captioned","art_coco_captioned"]:
        name=f"{embedding}-{data}"
        command=f"sbatch -J npz --err=slurm/npz_mini/{name}.err --out=slurm/npz_mini/{name}.out runpygpu.sh make_np_dataset.py  --embedding {embedding} "
        command+=f" --dataset jlbaker361/{data} --output_dataset jlbaker361/{name}-1000 --limit 1000 --rewrite "
        print(command)