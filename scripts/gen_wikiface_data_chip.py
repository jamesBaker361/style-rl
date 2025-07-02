for limit in [1000,5000,-1]:
    for pipeline in ["stability","lcm","sana"]:
        for embedding in ["siglip2","ssl","clip","dino"]:
            name=f"{embedding}-wikifaces{limit}-{pipeline}"
            command=f"sbatch -J npz --err=slurm_chip/npz/{name}.err --out=slurm_chip/npz/{name}.out runpygpu_chip.sh make_wikiart_dataset.py  --embedding {embedding} "
            command+=f"  --output_dataset jlbaker361/{name} --limit {limit} --rewrite --pipeline {pipeline} "
            print(command)