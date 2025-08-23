#https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/Annie/skins/skin22/images/annie_splash_uncentered_22.jpg
#/lol-game-data/assets/ASSETS/Characters/Annie/Skins/Skin22/Images/annie_splash_uncentered_22.jpg

import requests
from diffusers.utils.loading_utils import load_image
from datasets import Dataset, load_dataset
import time

url = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/skins.json"
response = requests.get(url)
import re
import PIL

failures=0
successes=0
count=0


if response.status_code==200:
    output_dict={
        "image":[],
        "url":[],
        "tag":[],
        "champion":[]
    }
    start=0
    try:
        old_dataset=load_dataset("jlbaker361/league-dragon-splash-tagged",split="train")
        output_dict=old_dataset.to_dict()
        start=len(output_dict["image"])
        print(f"skipping {start}")
    except:
        print("colu,ndt load from hf")

    bad_splash_list=[]

    data=response.json()
    for x,(k,v) in enumerate( data.items()):
        if x < start:
            continue
        uncenteredSplashPath=v["uncenteredSplashPath"]
        champion_start=uncenteredSplashPath.find("Characters/")+len("Characters/")
        champion_end=uncenteredSplashPath.find("/Skins")
        if champion_end==-1:
            champion_end=uncenteredSplashPath.find("/skins")
        champion=uncenteredSplashPath[champion_start:champion_end]
        tagged_name=v["name"].lower()
        tag=tagged_name[:tagged_name.find(champion.lower())]
        if tagged_name!=champion.lower() and tagged_name.find(champion.lower())==0:
            tag=tagged_name[:tagged_name.find(champion.lower())+len(champion)]
        is_base=v["isBase"]
        
        path=uncenteredSplashPath[champion_start:].lower().capitalize()
        first_num=int(re.search(r"\d+", path).group())
        #print(path,first_num,name)
        url=f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/{champion}/skins/skin{first_num}/images/{champion.lower()}_splash_uncentered_{first_num}.jpg"
        lower_url=f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/{champion.lower()}/skins/skin{first_num}/images/{champion.lower()}_splash_uncentered_{first_num}.jpg"
        weird_url=f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/{champion}/skins/skin{first_num}/images/{champion.lower()}_splash_uncentered_{first_num}.skins_{champion.lower()}_skin{first_num}.jpg"
        lower_weird_url=f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/{champion.lower()}/skins/skin{first_num}/images/{champion.lower()}_splash_uncentered_{first_num}.skins_{champion.lower()}_skin{first_num}.jpg"
        base_url=f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/{champion}/skins/base/images/{champion.lower()}_splash_uncentered_{first_num}.jpg"
        base_lower_url=f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/{champion.lower()}/skins/base/images/{champion.lower()}_splash_uncentered_{first_num}.jpg"
        first_num_padded="0"+str(first_num)
        url_zero=f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/{champion}/skins/skin{first_num_padded}/images/{champion.lower()}_splash_uncentered_{first_num}.jpg"
        lower_url_zero=f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/{champion.lower()}/skins/skin{first_num_padded}/images/{champion.lower()}_splash_uncentered_{first_num}.jpg"
        found=False     
        for candidate_url in [url,lower_url,weird_url,lower_weird_url,base_url,base_lower_url,url_zero,lower_url_zero]:
            try:
                output_dict["image"].append(load_image(candidate_url))
                output_dict["tag"].append(tag)
                output_dict["url"].append(candidate_url)
                output_dict["champion"].append(champion)
                found=True
                break
            except PIL.UnidentifiedImageError:
                pass
        if found==False:
            bad_splash_list.append({
                                        "id":k,
                                        "champion":champion,
                                        "tag":tag
                                    })

        
        if x % 5==0:
            time.sleep(10)
            Dataset.from_dict(output_dict).push_to_hub("jlbaker361/league-dragon-splash-tagged")
    print("bad splash",bad_splash_list)
    Dataset.from_dict(output_dict).push_to_hub("jlbaker361/league-dragon-splash-tagged")
        #the name could be capitalized or not capitalized and the number could  be 01 or 1
        #print(uncenteredSplashPath)