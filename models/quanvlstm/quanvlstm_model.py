import torch.nn as nn
from models.quanvlstm.quanvlstm import QuanvLSTM

class Predictor(nn.Module):
    def __init__(self, hidden_dim, output_dim, T_out):
        super().__init__()
        self.T_out = T_out
        self.conv = nn.Conv2d(hidden_dim, output_dim * T_out, kernel_size=3, padding=1)

    def forward(self, h):
        # h: (B, hidden_dim, H, W)
        out = self.conv(h)  # (B, output_dim * T_out, H, W)
        B, C, H, W = out.shape
        out = out.view(B, self.T_out, -1, H, W)  # (B, T_out, output_dim, H, W)
        return out


class QuanvLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, kernel_size_list, n_qubits_list, num_layers, output_dim, T_out):
        super().__init__()
        self.quanvlstm = QuanvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim_list,
            kernel_size=kernel_size_list,
            n_qubits_list=n_qubits_list,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False
        )
        self.predictor = Predictor(hidden_dim_list[-1], output_dim, T_out)

    def forward(self, x):
        layer_outputs, last_states = self.quanvlstm(x)
        last_hidden = last_states[-1][0]   # (B, hidden_dim, H, W)
        pred = self.predictor(last_hidden) # (B, T_out, output_dim, H, W)
        return pred