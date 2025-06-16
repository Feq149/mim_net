import torch
import sidechainnet as scn
import torch.optim as torch_optim
from torch.utils.tensorboard import SummaryWriter
import os

from models.mim_net import MimNet
from train.losses import design_loss, folding_loss, regularization_total_var
from train.utils import sanitize_batch
from train.utils import process_batch

if __name__ == "__main__":

    EPOCHS = 50  # liczba epok
    LR = 1e-4  # learning rate
    BETA = 1e-4  # waga regularizacji

    # Dane
    print("Ładowanie danych...")
    dataloaders = scn.load(
        casp_version=7, with_pytorch="dataloaders", batch_size=24
    )  # mały batch_size, żeby CUDA dała radę
    train_loader = dataloaders["train"]  # type: ignore
    num_samples = len(train_loader.dataset)  # type: ignore
    print("Dane załadowane.")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MimNet(in_channels=40, nf=128, T=6, n_levels=3).to(device)
    optimizer = torch_optim.Adam(model.parameters(), lr=LR)

    # Logi
    writer = SummaryWriter("logs/mimnet-train-without-regularization")  # type: ignore

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        fold_total = 0
        design_total = 0

        for i, batch in enumerate(train_loader):
            # batch = sanitize_batch(batch) # nie ma potrzeby jeśli pssm działa

            true_seq, true_3d, M = process_batch(batch, device)

            optimizer.zero_grad()

            expected_3d, expected_seq = model.forward_joint(true_seq)

            # Straty
            fold_loss = folding_loss(expected_3d, true_3d, M)
            des_loss = design_loss(expected_seq, true_seq)
            reg_loss = regularization_total_var(model.reversible.f_blocks)

            # Suma strat
            loss = fold_loss + des_loss + BETA * reg_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            fold_total += fold_loss.item()
            design_total += des_loss.item()

            if (i + 1) % 100 == 0:
                print(
                    f"[Epoch {epoch} | Batch {i + 1}] Loss: {loss.item():.4f} | Folding: {fold_loss.item():.4f} | Design: {des_loss.item():.4f}"
                )

        writer.add_scalar("Loss/train", total_loss / num_samples, epoch + 1)
        writer.add_scalar("Folding/train", fold_total / num_samples, epoch + 1)
        writer.add_scalar("Design/train", design_total / num_samples, epoch + 1)
        print(
            f"=====> Epoch {epoch} done. Loss: {total_loss / num_samples:.4f}, Folding: {fold_total / num_samples:.4f}, Design: {design_total / num_samples:.4f}"
        )  # type: ignore

    writer.close()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mimnet_model_without_reg.pth")
