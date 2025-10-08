from datasets import get_dataset_config_names,load_dataset,Dataset
import re

config_list = get_dataset_config_names("google/dreambooth")
print("Configs:", config_list)


data_dict={"image":[],"object":[]}

for config in config_list:
    if config=="default":
        continue
    ds=load_dataset("google/dreambooth",config)["train"]
    for row in ds:
        data_dict["image"].append(row["image"])
        data_dict["object"].append(re.sub(r"\d+", "", config).replace("_"," "))

Dataset.from_dict(data_dict).push_to_hub("jlbaker361/dreambooth")