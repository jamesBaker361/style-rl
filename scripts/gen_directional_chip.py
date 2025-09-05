for seed_offset in [x *10 for x in range(10)]:
    command=f"sbatch -J dir --err=slurm_chip/direction/{seed_offset}.err --out=slurm_chip/direction/{seed_offset}.out "
    command+=f" runpygpu_chip.sh directional_data.py --seed_offset {seed_offset} --dest_dataset jlbaker361/directional-{seed_offset} "
    command+=f" --upload_interval 50 "
    print(command)