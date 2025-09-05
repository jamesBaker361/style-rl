from datasets import Dataset
import requests
from PIL import Image
from io import BytesIO
import time

data_dict={
    "image":[],
    "image_url":[],
    "artist":[],
    "name":[],
    "type_line":[]
}



image_url_list=[]
for n in range(20):
    time.sleep(10)
    url = f"https://api.scryfall.com/cards/search?q=t:planeswalker loy={n}"
    req = requests.get(url).json()["data"]
    if req["status"]!=404:
        data=req["data"]

        for k,obj in enumerate(data):
            if "card_faces" in obj:
                for j,card in enumerate(obj["card_faces"]):
                    image_url=card["image_uris"]["art_crop"]
                    img_data = requests.get(image_url).content
                    image = Image.open(BytesIO(img_data))
                    artist=card["artist"]
                    name=card["name"]
                    type_line=card["type_line"]
            else:
                card=obj
                image_url=card["image_uris"]["art_crop"]
                img_data = requests.get(image_url).content
                image = Image.open(BytesIO(img_data))
                artist=card["artist"]
                name=card["name"]
                type_line=card["type_line"]
            data_dict["artist"].append(artist)
            data_dict["image"].append(image)
            data_dict["image_url"].append(image_url)
            data_dict["name"].append(name)
            data_dict["type_line"].append(type_line)

Dataset.from_dict(data_dict).push_to_hub("jlbaker361/mtg")

