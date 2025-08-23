import requests
from diffusers.utils.loading_utils import load_image
from datasets import Dataset
import time

url = "https://ddragon.leagueoflegends.com/cdn/15.16.1/data/en_US/champion.json"
response = requests.get(url)

if response.status_code == 200:
    output_dict={
        "image":[],
        "url":[],
        "tag":[],
        "champion":[]
    }
    data = response.json()
    names=[k for k in data["data"].keys()] 
    for n,name in enumerate(names):
        if n % 5==0:
            Dataset.from_dict(output_dict).push_to_hub("jlbaker361/league-splash-tagged")
            time.sleep(30)
        url_champion=f"https://ddragon.leagueoflegends.com/cdn/15.16.1/data/en_US/champion/{name}.json"
        response_champion = requests.get(url_champion)
        if response_champion.status_code==200:
            data_champion=response_champion.json()
            skin_list=data_champion["data"][name]["skins"]
            for skin in skin_list:
                skin_num=skin["num"]
                url_skin=f"https://ddragon.leagueoflegends.com/cdn/img/champion/splash/{name}_{skin_num}.jpg"
                skin_name=skin["name"]
                if skin_name=="default":
                    tag=""
                else:
                    index=skin_name.find(name)
                    tag=skin_name[:index-1]
                output_dict["champion"].append(name)
                output_dict["tag"].append(tag)
                output_dict["url"].append(url_skin)
                output_dict["image"].append(load_image(url_skin))
    Dataset.from_dict(output_dict).push_to_hub("jlbaker361/league-splash-tagged")
else:
    print(f"Error fetching data: {response.status_code}")
