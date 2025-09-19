for limit in [1000,5000,-1]:
    for pipeline in ["stability","lcm","sana"]:
        for embedding in ["siglip2","ssl","clip","dino"]:
            for data in ['wikiart',"realkaggle"]:
                name=f"{embedding}-{data}{limit}-{pipeline}"
                command=f"sbatch -J npz --err=slurm_chip/npz/{name}.err --out=slurm_chip/npz/{name}.out runpygpu_chip.sh make_wikiart_dataset.py  --embedding {embedding} "
                command+=f"  --output_dataset jlbaker361/{name} --limit {limit}  --pipeline {pipeline} "
                dataset_choice={
                    "wikiart":" asahi417/wikiart-face ",
                    "realkaggle":"kaustubhdhote/human-faces-dataset"
                }[data]
                command+=f" --dataset {dataset_choice}"
                print(command)