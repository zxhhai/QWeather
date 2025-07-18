from models.convlstm import ConvLSTM
from datasets.dataset import WeatherDataset
from utils.trainer import Trainer
from utils.optimizer import get_optimizer, get_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


import torch.nn as nn

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


class ConvLSTMWeatherModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, kernel_size_list, num_layers, output_dim, T_out):
        super().__init__()
        # 这里实例化自己的ConvLSTM
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim_list,
            kernel_size=kernel_size_list,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        # Predictor的hidden_dim应该是最后一层的hidden_dim
        self.predictor = Predictor(hidden_dim_list[-1], output_dim, T_out)

    def forward(self, x):
        layer_outputs, last_states = self.convlstm(x)
        last_hidden = last_states[-1][0]  # 取最后一层最后时间步的隐藏状态 (B, hidden_dim, H, W)
        pred = self.predictor(last_hidden)  # (B, T_out, output_dim, H, W)
        return pred


# 示例调用
model = ConvLSTMWeatherModel(
    input_dim=4,
    hidden_dim_list=[16, 16, 4],
    kernel_size_list=[(3, 3), (3, 3), (3, 3)],
    num_layers=3,
    output_dim=4,
    T_out=1
)


optimizer = get_optimizer(model, 'adam', lr=0.001)
scheduler = get_scheduler(optimizer, 'step', step_size=10, gamma=0.1)

trainer = Trainer(
    model=model,
    criterion=nn.MSELoss(),
    optimizer=optimizer,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    scheduler=scheduler,
    save_path='./checkpoints',
)

train_dataset = WeatherDataset(
    file_path='/home/zxh/CQ/QWeather/testdata/test_data.nc',
    input_seq_len=8,
    target_seq_len=1,
    var_name='data'
)
val_dataset = WeatherDataset(
    file_path='/home/zxh/CQ/QWeather/testdata/test_data.nc',
    input_seq_len=8,
    target_seq_len=1,
    var_name='data'
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)


trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
)
