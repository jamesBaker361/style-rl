for pipeline in ["DiffusionPipeline","CompatibleLatentConsistencyModelPipeline"]:
    for scheduler in ["LCMScheduler","DDIMScheduler","DEISMultistepScheduler"]:
        name=f"{pipeline}_{scheduler}"
        command=f"sbatch -J ip --err=slurm_chip/ip/{name}.err --out=slurm_chip/ip/{name}.out runpygpu_chip.sh ip_test.py --scheduler_type {scheduler} --pipeline_type {pipeline} "
        print(command)