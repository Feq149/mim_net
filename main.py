import torch
import sidechainnet as scn
import torch.optim as torch_optim

from models.mim_net import MimNet
from train.losses import design_loss, folding_loss, regularization_total_var


def sanitize_batch(batch):
    # Napraw dane zawierające NaN — zastąp 0
    batch.evolutionary = torch.nan_to_num(batch.evolutionary, nan=0.0)
    batch.coords = torch.nan_to_num(batch.coords, nan=0.0)
    return batch


if __name__ == "__main__":

    EPOCHS = 10
    LR = 1e-4
    BETA = 1e-4  # waga regularizacji

    # Dane
    print("Ładowanie danych")
    dataloaders = scn.load(casp_version=12, with_pytorch="dataloaders",casp_thinning="scnmin",)
    train_loader = dataloaders['train']
    print("Dane załadowane.")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MimNet(in_channels=40, nf=128, T=6, n_levels=3).to(device)
    optimizer = torch_optim.Adam(model.parameters(), lr=LR)

    # Trening
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        fold_total = 0
        design_total = 0

        for i, batch in enumerate(train_loader):
            #batch = sanitize_batch(batch)
            one_hot_encoding = batch.seqs_onehot.permute(0, 2, 1).float().to(device)  # (B, 20, L)
            pssm = torch.nan_to_num(batch.evolutionary, nan=0.0)
            coords = torch.nan_to_num(batch.coords, nan=0.0)
            if pssm.shape[2] < 20:
                pad_width = 20 - pssm.shape[2]
                padding = torch.zeros(pssm.shape[0], pssm.shape[1], pad_width, device=pssm.device)
                pssm = torch.cat([pssm, padding], dim=2)
            else:
                pssm = pssm[:, :, :20]

            pssm = pssm.permute(0, 2, 1).float().to(device)  # (B, 20, L)
            true_seq = torch.cat([one_hot_encoding, pssm], dim=1)  # (B, 40, L)
            coords_ca = coords[:, :, 1, :].float().to(device)  # (B, L, 3)
            true_3d = coords_ca.permute(0, 2, 1).contiguous()  # (B, 3, L)


            #print("S_true mean:", S_true.mean().item())
            #print("X_true mean:", X_true.mean().item())
            mask = batch.masks.float().to(device)  # (B, L)


            def print_nan_stats(name, tensor):
                print(
                    f"{name}: has_nan={torch.isnan(tensor).any().item()}, shape={tensor.shape}, min={tensor.min().item() if not torch.isnan(tensor).any() else 'NaN'}, max={tensor.max().item() if not torch.isnan(tensor).any() else 'NaN'}")


            # print_nan_stats("one_hot", one_hot)
            # print_nan_stats("pssm", pssm)
            # print_nan_stats("coords_ca", coords_ca)
            # print_nan_stats("mask", mask)
            # exit(0)
            M = mask.unsqueeze(1) * mask.unsqueeze(2)#bo część danych jest nan lub zera lun dziury, broadcasting pytorcha in action
            #print("M sum:", M.sum().item())

            optimizer.zero_grad()

            expected_3d, expected_seq = model.forward_joint(true_seq)


            # Straty
            fold_loss = folding_loss(expected_3d, true_3d, M)
            des_loss = design_loss(expected_seq, true_seq)
            rev_params = [p for p in model.reversible.parameters() if p.requires_grad]
            reg_loss = regularization_total_var(rev_params)

            # Suma strat
            loss = fold_loss + des_loss + BETA * reg_loss
            # try:
            #     with torch.autograd.detect_anomaly():
            loss.backward()
            # except Exception as e:
            #     print("Błąd podczas backward:", e)
            #     exit(0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)



            optimizer.step()

            total_loss += loss.item()
            fold_total += fold_loss.item()
            #print("fold_loss:", fold_loss.item())
            #print("design_loss:", des_loss.item())

            design_total += des_loss.item()

            if (i + 1) % 100 == 0:
                print(
                    f"[Epoch {epoch} | Batch {i + 1}] Loss: {loss.item():.4f} | Folding: {fold_loss.item():.4f} | Design: {des_loss.item():.4f}")

        avg_loss = total_loss / (i + 1)
        print(
            f"=====> Epoch {epoch} done. Avg Loss: {avg_loss:.4f}, Folding: {fold_total / (i + 1):.4f}, Design: {design_total / (i + 1):.4f}")
