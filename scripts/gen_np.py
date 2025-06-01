for embedding in ["clip","ssl","siglip2","dino"]:
    for data in ["league_captioned_tile","league_captioned_splash","coco_captioned","art_coco_captioned","celeb_captioned"]:
        name=f"{embedding}-{data}"
        command=f"sbatch -J npz --err=slurm/npz/{name}.err --out=slurm/npz/{name}.out runpygpu.sh make_np_dataset.py  --embedding {embedding} "
        command+=f" --dataset jlbaker361/{data} --output_dataset jlbaker361/{name} --limit -1"
        print(command)