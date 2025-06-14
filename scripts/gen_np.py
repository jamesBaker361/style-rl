for embedding in ["siglip2","ssl","clip","dino"]:
    for data in ["league_captioned_splash","coco_captioned","art_coco_captioned"]:
        for pipeline in ["sana"]:
            for limit in [50]:
                name=f"{embedding}-{data}-{limit}-{pipeline}"
                command=f"sbatch -J npz --err=slurm/npz/{name}.err --out=slurm/npz/{name}.out runpygpu.sh make_np_dataset.py  --embedding {embedding} "
                command+=f" --dataset jlbaker361/{data} --output_dataset jlbaker361/{name} --limit {limit}  --pipeline {pipeline} --rewrite --mixed_precision no "
                print(command)