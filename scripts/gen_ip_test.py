for pipeline in ["DiffusionPipeline","CompatibleLatentConsistencyModelPipeline"]:
    for scheduler in ["LCMScheduler","DDIMScheduler","DEISMultistepScheduler"]:
        name=f"{pipeline}_{scheduler}"
        command=f"sbatch -J ip --err=slurm/ip/{name}.err --out=slurm/ip/{name}.out runpygpu.sh ip_test.py --scheduler_type {scheduler} --pipeline_type {pipeline} "
        print(command)