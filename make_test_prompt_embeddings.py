from datasets import Dataset

from pipelines import CompatibleLatentConsistencyModelPipeline

data_dict={
    "prompt":[],
    "text_embedding":[]
}

pipe=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to("cuda")

with open("test_prompts.txt","r") as file:
    for line in file:
        line=line.strip()
        data_dict["prompt"].append(line)
        text_embedding,_=pipe.encode_prompt(line,
                                        "cuda", #accelerator.device,
                                        1,
                                        pipe.do_classifier_free_guidance,
                                        negative_prompt=None,
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )
        data_dict["text_embedding"].append(text_embedding.cpu().detach().numpy())

Dataset.from_dict(data_dict).push_to_hub("jlbaker361/test_prompts")