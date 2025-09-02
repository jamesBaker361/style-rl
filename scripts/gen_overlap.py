for overlap_frac in [0.25,0.5,0.75]:
    for kv_type in ["str","ip"]:
        for initial_ip_adapter_scale in [0.5,0.75]:
            name=f"{kv_type}_{overlap_frac}_{initial_ip_adapter_scale}"
            command=f"sbatch --exclude=gpu[005,006,008,010,011,013,014,018],cuda[001-008],pascal[006-010],gpuk[001-012] --err=slurm/overlap/_{name}.err --out=slurm/overlap/_{name}.out runpygpu.sh main_seg.py --segmentation_attention_method overlap --initial_mask_step_list 1 2 --final_mask_steps_list 2 3 4 5 --final_adapter_steps_list 2 3 4 5 --limit 16 --overlap_frac {overlap_frac} "
            command+=f" --kv_type {kv_type} "
            command+=f" --initial_ip_adapter_scale {initial_ip_adapter_scale} "
            print(command)