for overlap_frac in [0.5,0.75,0.25]:
    for kv_type in ["ip"]:
        for initial_ip_adapter_scale in [0.5,0.75]:
            for step_set in ["mid","late","early"]:
                for limited in ["yes","no"]:
                    for data in ["sshq-100","mtg"]:
                        name=f"{kv_type}_{overlap_frac}_{initial_ip_adapter_scale}_{step_set}_{limited}_{data}"
                        step_dict={
                            "early": " 0 1 2 3",
                            "mid": " 2 3 4 5",
                            "late": " 4 5 6 7"
                        }
                        final_mask_steps_list=step_dict[step_set]
                        if limited=="yes":
                            final_adapter_steps_list=step_dict[step_set]
                        else:
                            final_adapter_steps_list=" 0 1 2 3 4 5 6 7 "
                        command=f"sbatch -J seg --exclude=gpu[005,006,008,010,011,013,014,018],cuda[001-008],pascal[006-010],gpuk[001-012] --err=slurm/overlap/_{name}.err --out=slurm/overlap/_{name}.out runpygpu.sh main_seg.py --segmentation_attention_method overlap --initial_mask_step_list 1 2 --final_mask_steps_list {final_mask_steps_list} --final_adapter_steps_list {final_adapter_steps_list} --limit 32 --overlap_frac {overlap_frac} "
                        command+=f" --kv_type {kv_type} "
                        if kv_type=="str":
                            command+=f" --token 0 "
                        command+=f" --project_name seg-{data} "
                        command+=f" --initial_ip_adapter_scale {initial_ip_adapter_scale} "
                        command+=f"  --dataset jlbaker361/{data} "
                        print(command)