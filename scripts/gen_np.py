for embedding in ["siglip2","ssl","clip","dino"]:
    for data in ["league_captioned_splash"]:
        for pipeline in ["lcm"]:
            for limit in [1000,250]:
                name=f"{embedding}-{data}-{limit}-{pipeline}"
                command=f"sbatch -J npz --err=slurm/npz/{name}.err --out=slurm/npz/{name}.out runpygpu.sh make_np_dataset.py  --embedding {embedding} "
                command+=f" --dataset jlbaker361/{data} --output_dataset jlbaker361/{name} --limit {limit}  --pipeline {pipeline} --rewrite --mixed_precision no "
                print(command)