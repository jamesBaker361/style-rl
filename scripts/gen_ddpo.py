for start in [0]:
    limit=start+4
    for version in ["style","style2" ]: #,"content","baseline","both"]:
        flags={
            "style2":" --style_layers_train --style_layers 0 1 2 ",
            "style":" --style_layers_train --style_layers 0 1 ",
            "content":" --content_layers_train ",
            "both":" --style_layers_train --style_layers 0 1  --content_layers_train ",
            "baseline":"  --style_layers_train --style_layers 0 1 2 3  --style_mid_block "
        }[version]
        for lr in [0.001,0.01,0.1]:
            exclude=" --exclude=gpu[005,006,010-014,017,018],cuda[001-008],pascal[001-010] " #gpu[011-014,017-018,021,027-028,005-010],cuda[001-008]
            command=f" sbatch -J ddpo {exclude} --err=slurm/ddpo/{version}_{start}_{lr}.err --out=slurm/ddpo/{version}_{start}_{lr}.out runpygpu.sh main.py --batch_size 1 --gradient_accumulation_steps 2 "
            command+=f" --start {start} --limit {limit} {flags} --mixed_precision no --sample_num_batches_per_epoch 16 --image_size 256 --epochs 50"
            command+=f" --learning_rate {lr} "
            print(command)