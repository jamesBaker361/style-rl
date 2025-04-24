exclude=" --exclude=gpu[005,006,008,010-014,017,018],cuda[001-008],pascal[001-010] " 
for lr in [0.001,0.01]:
    for reward_fn in ["ir"]:
        for prompt_src in ["animals","nature"]:
            for method in ["unet", "pp","ti"]:
                for submethod in ["vanilla","nemesis"]:
                    name=f"{reward_fn}_{lr}_{reward_fn}_{prompt_src}_{method}_{submethod}"
                    method_flags={
                        "unet":" --train_unet --style_layers 0 1 2 3 ",
                        "pp":" --textual_inversion --use_pplus ",
                        "ti": " --textual_inversion "
                    }[method]
                    submethod_flags={
                        "vanilla":"",
                        "nemesis":" --nemesis "
                    }[submethod]
                    src_text={
                            "animals":"random_animals.txt",
                            "nature":"nature_prompts.txt"
                    }[prompt_src]
                    command=f"sbatch -J {reward_fn}-{prompt_src}-ti-3000 --err=slurm/{prompt_src}3000/{name}.err --out=slurm/{prompt_src}3000/{name}.out"
                    command+=f" {exclude} runpygpu.sh main_ir.py    {method_flags} {submethod_flags} --num_vectors 1 "
                    command+=f"   --prompt_src_txt {src_text} "
                    command+=f" --validation_epochs 40 --learning_rate {lr} --reward_fn {reward_fn} "
                    #command+=f" --project_name testing --per_prompt_stat_tracking_buffer_size 3 --epochs 10"
                    command+=f"  --project_name {reward_fn}_{prompt_src}_3000 --epochs 3000 "
                    print(command)
                    