sbatch -J pix --constraint=L40S --err=slurm_chip/eval/pix_mtg.err --out=slurm_chip/eval/pix_mtg.out runpygpu_chip.sh instruct.py --src_dataset jlbaker361/mtg --dest_dataset jlbaker361/pix-mtg --model pix2pix 
sbatch -J pix --constraint=L40S --err=slurm_chip/eval/pix_dreambooth.err --out=slurm_chip/eval/pix_dreambooth.out runpygpu_chip.sh instruct.py --src_dataset jlbaker361/dreambooth --dest_dataset jlbaker361/pix-dreambooth --model pix2pix 

sbatch -J clip --constraint=L40S --err=slurm_chip/eval/clip_mtg.err --out=slurm_chip/eval/clip_mtg.out runpygpu_chip.sh instruct.py --src_dataset jlbaker361/mtg --dest_dataset jlbaker361/instruct_clip-mtg --model instruct_clip 
sbatch -J clip --constraint=L40S --err=slurm_chip/eval/clip_dreambooth.err --out=slurm_chip/eval/clip_dreambooth.out runpygpu_chip.sh instruct.py --src_dataset jlbaker361/dreambooth --dest_dataset jlbaker361/instruct_clip-dreambooth --model instruct_clip

sbatch -J ultra --constraint=L40S --err=slurm_chip/eval/ultra_edit_mtg.err --out=slurm_chip/eval/ultra_edit_mtg.out runpygpu_chip.sh instruct.py --src_dataset jlbaker361/mtg --dest_dataset jlbaker361/ultra_edit-mtg --model ultra_edit
sbatch -J ultra --constraint=L40S --err=slurm_chip/eval/ultra_edit_dreambooth.err --out=slurm_chip/eval/ultra_edit_dreambooth.out runpygpu_chip.sh instruct.py --src_dataset jlbaker361/dreambooth --dest_dataset jlbaker361/ultra_edit-dreambooth --model ultra_edit

sbatch -J attn --constraint=L40S --err=slurm_chip/eval/attn_mtg.err --out=slurm_chip/eval/attn_mtg.out runpygpu_chip.sh main_seg.py --src_dataset jlbaker361/mtg --dest_dataset jlbaker361/attn-mtg
sbatch -J attn --constraint=L40S --err=slurm_chip/eval/attn_dreambooth.err --out=slurm_chip/eval/attn_dreambooth.out runpygpu_chip.sh main_seg.py --src_dataset jlbaker361/dreambooth --dest_dataset jlbaker361/attn-dreambooth