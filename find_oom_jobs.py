import os

cwd=os.getcwd()


cluster="amarel"
if cwd.find("donengel")!=-1:
    cluster="chip"

slurm_folder={
    "amarel":"slurm",
    "chip":"slurm_chip"
}[cluster]

dataset_list={
    "amarel":["pers2_league_captioned_splash",],
    "chip":["pers_art_coco_captioned"]
}[cluster]

target_job_list=[]

for dataset in dataset_list:
    for root, dirs, files in os.walk(os.path.join(slurm_folder,dataset)):
        for name in files:
            if name.endswith("out"):
                found=False
                with open(os.path.join(root,name),"r") as read_file:
                    for line in read_file:
                        if line.startswith("OOM"):
                            target_job_list.append(name)
                            found=True
                '''if found==False:
                    err_file=name.replace("out","err")
                    srun_error=False
                    with open(os.path.join(root,err_file),"r",errors="ignore") as read_file:
                        for line in read_file:
                            if line.find("srun: error")!=-1:
                                srun_error=True
                    if srun_error:
                        target_job_list.append(name)'''

command_list=[]

for root, dirs, files in os.walk("scripts"):
    for name in files:
        if name.endswith("sh"):
            with open(os.path.join(root,name),"r") as read_file:
                for line in read_file:
                    for target_job in target_job_list:
                        if line.find(target_job)!=-1:
                            command_list.append(line)

for command in command_list:
    print(command)