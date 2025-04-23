exclude=" --exclude=gpu[005,006,008,010-014,017,018],cuda[001-008],pascal[001-010] " 
for lr in [0.001,0.01]:
    for unet in ["no-unet","unet"]:
        for pplus in ["no-pplus","pplus"]:
            for num_vectors in [1]:
                for reward_fn in ["qualiclip","ir"]:
                    for prompt_src in ["animals","nature"]:

                        name=f"ti_{prompt_src}_{reward_fn}_{lr}_{unet}_{pplus}_{num_vectors}"
                        unet_flag={
                            "no-unet":"",
                            "unet":"--train_unet"
                        }[unet]
                        pplus_flag={
                            "no-pplus":"",
                            "pplus": "--use_pplus"
                        }[pplus]
                        src_text={
                            "animals":"random_animals.txt",
                            "nature":"nature_prompts.txt"
                        }[prompt_src]
                        command=f"sbatch -J {reward_fn}-{prompt_src}-ti --err=slurm/ti/{name}.err --out=slurm/ti/{name}.out"
                        command+=f" {exclude} runpygpu.sh main_ir.py  --epochs 10000  {unet_flag} {pplus_flag} --num_vectors {num_vectors}"
                        command+=f"   --project_name {reward_fn} --textual_inversion --prompt_src_txt {src_text} "
                        command+=f" --validation_epochs 40 --learning_rate {lr} --reward_fn {reward_fn} "
                        print(command)