sbatch -J pix  --err=slurm/eval/pix_mtg.err --out=slurm/eval/pix_mtg.out runpygpu.sh instruct.py --src_dataset jlbaker361/mtg --dest_dataset jlbaker361/pix-mtg --model pix2pix 
sbatch -J pix  --err=slurm/eval/pix_dreambooth.err --out=slurm/eval/pix_dreambooth.out runpygpu.sh instruct.py --src_dataset jlbaker361/dreambooth --dest_dataset jlbaker361/pix-dreambooth --model pix2pix 

sbatch -J clip --err=slurm/eval/clip_mtg.err --out=slurm/eval/clip_mtg.out runpygpu.sh instruct.py --src_dataset jlbaker361/mtg --dest_dataset jlbaker361/instruct_clip-mtg --model instruct_clip 
sbatch -J clip  --err=slurm/eval/clip_dreambooth.err --out=slurm/eval/clip_dreambooth.out runpygpu.sh instruct.py --src_dataset jlbaker361/dreambooth --dest_dataset jlbaker361/instruct_clip-dreambooth --model instruct_clip

sbatch -J ultra --err=slurm/eval/ultra_edit_mtg.err --out=slurm/eval/ultra_edit_mtg.out runpygpu.sh instruct.py --src_dataset jlbaker361/mtg --dest_dataset jlbaker361/ultra_edit-mtg --model ultra_edit
sbatch -J ultra  --err=slurm/eval/ultra_edit_dreambooth.err --out=slurm/eval/ultra_edit_dreambooth.out runpygpu.sh instruct.py --src_dataset jlbaker361/dreambooth --dest_dataset jlbaker361/ultra_edit-dreambooth --model ultra_edit

sbatch -J attn  --err=slurm/eval/attn_mtg.err --out=slurm/eval/attn_mtg.out runpygpu.sh main_seg.py --src_dataset jlbaker361/mtg --dest_dataset jlbaker361/attn-mtg
sbatch -J attn  --err=slurm/eval/attn_dreambooth.err --out=slurm/eval/attn_dreambooth.out runpygpu.sh main_seg.py --src_dataset jlbaker361/dreambooth --dest_dataset jlbaker361/attn-dreambooth