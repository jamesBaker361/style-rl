exclude=" --exclude=gpu[005,006,008,010-014,017,018],cuda[001-008],pascal[001-010] " 
#for lr in [0.001]:
for pipeline in ["lcm"]:
    for embedding in ["ssl"]:
        for training in ["reward","epsilon","v_prediction"]:
            for data in ["league_captioned_tile","league_captioned_splash","coco_captioned","art_coco_captioned","celeb_captioned"]:
                name=f"{pipeline}_{embedding}_{data}_{training}"
                training_flags={
                    "reward":" --training_type reward",
                    "epsilon":" --training_type denoise  --prediction_type epsilon",
                    "v_prediction":" --training_type denoise --prediction_type v_prediction "
                }[training]
                precision={
                    "lcm":"bf16"
                }[pipeline]
                command+=f" sbatch -J pers {exclude} --out=slurm/pers/{name}.out --err=slurm/pers/{name}.err "
                command+=f" runpygpu.sh main_pers.py  --mixed_precision {precision} --gradient_accumulation_steps 32 --embedding {embedding} --epochs 100 "
                command+=f"  {training_flags}  --validation_interval 100 --dataset jlbaker361/{data} --project_name pers_{data} "
                command+=f" --pipeline {pipeline} "


'''
parser.add_argument("--dataset",type=str,default="jlbaker361/captioned-images")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--embedding",type=str,default="dino")
parser.add_argument("--embedding",type=str,default="dino")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--data_dir",type=str,default="data_dir")
parser.add_argument("--save_data_npz",action="store_true")
parser.add_argument("--load_data_npz",action="store_true")
parser.add_argument("--pipeline",type=str,default="lcm")
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--training_type",help="denoise or reward",default="denoise")
parser.add_argument("--train_unet",action="store_true")
parser.add_argument("--prediction_type",type=str,default="epsilon")
parser.add_argument("--train_split",type=float,default=0.96)
parser.add_argument("--validation_interval",type=int,default=20)
parser.add_argument("--buffer_size",type=int,default=0)
parser.add_argument("--uncaptioned_frac",type=float,default=0.75)
parser.add_argument("--cross_attention_dim",type=int,default=1024)
parser.add_argument("--limit",type=int,default=-1)
'''