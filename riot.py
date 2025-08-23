import requests
from diffusers.utils.loading_utils import load_image
from datasets import Dataset,load_dataset
import time
import PIL

start=time.time()
url = "https://ddragon.leagueoflegends.com/cdn/15.16.1/data/en_US/champion.json"
response = requests.get(url)

failures=0
successes=0
count=0

if response.status_code == 200:
    output_dict={
        "image":[],
        "url":[],
        "tag":[],
        "champion":[]
    }
    start=0
    try:
        old_dataset=load_dataset("jlbaker361/league-splash-tagged",split="train")
        output_dict=old_dataset.to_dict()
        start=len(output_dict["image"])
        print(f"skipping {start}")
    except:
        print("colu,ndt load from hf")
    data = response.json()
    names=[k for k in data["data"].keys()]
    for n,name in enumerate(names):
        

        champion_id=data["data"][name]["id"]
        if n % 5==0:
            Dataset.from_dict(output_dict).push_to_hub("jlbaker361/league-splash-tagged")
            time.sleep(10)
        url_champion=f"https://ddragon.leagueoflegends.com/cdn/15.16.1/data/en_US/champion/{champion_id}.json"
        #print(url_champion)
        response_champion = requests.get(url_champion)
        if response_champion.status_code==200:
            data_champion=response_champion.json()
            skin_list=data_champion["data"][champion_id]["skins"]
            for skin in skin_list:
                if count < start:
                    continue
                skin_num=skin["num"]
                url_skin=f"https://ddragon.leagueoflegends.com/cdn/img/champion/splash/{champion_id}_{skin_num}.jpg"
                #print(url_skin)
                skin_name=skin["name"]
                if skin_name=="default":
                    tag=""
                else:
                    index=skin_name.find(champion_id)
                    tag=skin_name[:index-1]

                try:
                    output_dict["image"].append(load_image(url_skin))
                    output_dict["champion"].append(champion_id)
                    output_dict["tag"].append(tag)
                    output_dict["url"].append(url_skin)
                    successes+=1
                except PIL.UnidentifiedImageError:
                    print("bad url!", url_skin)
                    failures+=1
                count+=1 
    Dataset.from_dict(output_dict).push_to_hub("jlbaker361/league-splash-tagged")
    end=time.time()
    print(f"all done! failures: {failures} successes: {successes} elpased {end-start}")
else:
    print(f"Error fetching data: {response.status_code}")
