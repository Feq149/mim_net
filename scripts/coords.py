import torch
import sidechainnet as scn
from models.mim_net import MimNet
from scripts.visualize_3d import plot_real_vs_predicted

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloaders = scn.load(casp_version=7, with_pytorch="dataloaders", batch_size=1)
test_loader = dataloaders["train"]  # type: ignore

model = MimNet(in_channels=40, nf=128, T=6, n_levels=3).to(device)
checkpoint_path = "checkpoints/mimnet_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Save images to files instead of displaying
output_dir = "images"
import os
os.makedirs(output_dir, exist_ok=True)
plot_real_vs_predicted(
    model=model,
    dataloader=test_loader,
    device=device,
    save_path=os.path.join(output_dir, "coords.png")
)
