from datasets import load_dataset

for data in ["league_captioned_tile","league_captioned_splash","coco_captioned","art_coco_captioned","celeb_captioned"]:
    load_dataset(f"jlbaker361/{data}",split="train",force_download=True)