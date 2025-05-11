exclude=" --exclude=gpu[005,006,008,010-014,017,018],cuda[001-008],pascal[001-010] " 
for lr in [0.01]:
    for start in [0]:
        limit=start+4
        for version in ["baseline" ]: #,"content","baseline","both"]:
            flags={
                "style2":" --style_layers_train --style_layers 0 1 2 ",
                "style":" --style_layers_train --style_layers 0 1 ",
                "content":" --style_layers_train  --style_layers 0 1 --style_dataset jlbaker361/people --prompt portrait --project_name content",
                "both":" --style_layers_train --style_layers 0 1  --content_layers_train ",
                "baseline":"  --style_layers_train --style_layers 0 1 2 3  --style_mid_block ",
                "upper":"--style_layers_train --style_layers 2 3  --style_mid_block"
            }[version]
            for epochs in [250,100]:
                for reward_fn in ["vgg"]:
                    for version in [19,16]:
                        layers={
                            19:" --vgg_layer_indices 4 9 18 ",
                            16:" --vgg_layer_indices 4 9 16 "
                        }[version]
                        name=f"{version}_{start}_{lr}_{epochs}_{reward_fn}_{version}_prompt"
                        command=f"sbatch -J style --err=slurm/style_vgg/{name}.err --out=slurm/style_vgg/{name}.out"
                        command+=f" {exclude} runpygpu.sh main.py --start {start} --limit {limit} --reward_fn {reward_fn} --epochs {epochs} {flags}"
                        command+=f" --content_limit 1 --prompt_alignment {layers} --vgg_n {version} "
                        print(command)