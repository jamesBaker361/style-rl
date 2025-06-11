for embedding in ["siglip2","ssl","clip","dino"]:
    for data in ["league_captioned_splash","coco_captioned","art_coco_captioned"]:
        for pipeline in ["sana"]:
            for limit in [1000,-1]:
                name=f"{embedding}-{data}-{limit}-{pipeline}"
                command=f"sbatch -J npz --err=slurm/npz_mini/{name}.err --out=slurm/npz_mini/{name}.out runpygpu.sh make_np_dataset.py  --embedding {embedding} "
                command+=f" --dataset jlbaker361/{data} --output_dataset jlbaker361/{name}-{limit} --limit {limit} --rewrite --pipeline {pipeline} "
                print(command)