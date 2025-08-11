import torch
import torch.nn as nn
import time
import xarray as xr

class Evaluater:
    def __init__(self, model_class, model_kwargs, model_path, dataloader, device):
        self.device = device
        self.dataloader = dataloader

        self.model = model_class(**model_kwargs).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def evaluate_mse(self, save_path="./quanvlstm_output.nc"):
        mse_loss_fn = nn.MSELoss(reduction='mean')
        total_loss = 0.0
        total_samples = 0
        predictions_list = []

        start_time = time.time()
        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)  # [B, C, H, W]
                loss = mse_loss_fn(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                predictions_list.append(outputs.cpu())

                avg_mse = total_loss / total_samples
                print({"MSE": f"{avg_mse:.6f}"})

        avg_mse = total_loss / total_samples
        elapsed_time = time.time() - start_time
        print(f"Final MSE: {avg_mse:.6f} | Time: {elapsed_time:.2f}s")

        predictions = torch.cat(predictions_list, dim=0)  # 可能是 [N, T, C, H, W]

        if predictions.ndim == 5:
            # 如果有时间维度，可以选择最后一个时间步
            predictions = predictions[:, -1, :, :, :]  # 取最后一帧
            # 或 predictions = predictions.reshape(-1, *predictions.shape[2:])  # 展平 N*T

        elif predictions.ndim != 4:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")

        predictions = predictions.numpy()

        N, C, H, W = predictions.shape
        ds = xr.DataArray(
            predictions,
            dims=("sample", "channel", "height", "width"),
            coords={
                "sample": range(N),
                "channel": range(C),
                "height": range(H),
                "width": range(W)
            },
            name="predictions"
        ).to_dataset()

        ds.to_netcdf(save_path)
        print(f"Saved predictions to {save_path}")

        return avg_mse
