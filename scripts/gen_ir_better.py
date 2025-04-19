exclude=" --exclude=gpu[005,006,008,010-014,017,018],cuda[001-008],pascal[001-010] " 
for lr in [0.001,0.0001]:
    for unet in ["no-unet","unet"]:
        for pplus in ["no-pplus","pplus"]:
            for num_vectors in [1,3]:
                name=f"ti_{lr}_{unet}_{pplus}_{num_vectors}"
                unet_flag={
                    "no-unet":"",
                    "unet":"--train_unet"
                }[unet]
                pplus_flag={
                    "no-pplus":"",
                    "pplus": "--use_pplus"
                }[pplus]
                command=f"sbatch -J ir-ti --err=slurm/ir/{name}.err --out=slurm/ir/{name}.out"
                command+=f" {exclude} runpygpu.sh main_ir.py  --epochs 1000  {unet_flag} {pplus_flag} --num_vectors {num_vectors}"
                command+=f"   --project_name ti-ir --textual_inversion --prompt_src_txt random_animals.txt"
                command+=f" --validation_epochs 40 --learning_rate {lr}"
                print(command)