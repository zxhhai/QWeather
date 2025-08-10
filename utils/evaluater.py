import torch
import torch.nn as nn
import time

class Evaluater:
    def __init__(self, model_class, model_kwargs, model_path, dataloader, device):
        self.device = device
        self.dataloader = dataloader

        self.model = model_class(**model_kwargs).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def evaluate_mse(self):
        mse_loss_fn = nn.MSELoss(reduction='mean')
        total_loss = 0.0
        total_samples = 0

        start_time = time.time()
        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = mse_loss_fn(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                avg_mse = total_loss / total_samples
                print({"MSE": f"{avg_mse:.6f}"})

        avg_mse = total_loss / total_samples
        elapsed_time = time.time() - start_time

        print(f"final MSE: {avg_mse:.6f} | Time: {elapsed_time:.2f}s")
        return avg_mse