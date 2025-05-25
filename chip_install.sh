python -m venv myenv
source myenv/bin/activate
pip install accelerate torch torchmetrics torchvision diffusers trl["diffusers"] transformers wandb datasets
python --version