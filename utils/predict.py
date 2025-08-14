import torch
from models.quanvlstm.quanvlstm_model import QuanvLSTMModel 
from datasets.dataset import create_dataloader
from utils.evaluater import Evaluater
import xarray as xr
import numpy as np
import pickle

device = "cuda"

model = QuanvLSTMModel(
    input_dim=10,
    hidden_dim_list=[32, 16],
    kernel_size_list=[(3, 3), (3, 3)],
    num_layers=2,
    output_dim=10,
    T_out=1,
    n_qubits_list=[4, 4]
).to(device)

model_path="./pretrained/quanvlstm.pth" # 待评估模型保存路径
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

from datasets.dataset import WeatherDataset

# 参数
data_file = "./datasets/data/data_large.nc"       # 原始数据文件路径
scaler_file = "./datasets/scaler/quanvlstm_scaler.pkl"     # 训练时保存的scaler
var_name=["chla", "hcho", "no2", "o3", "par", "sla", "sst", "wind", "DML", "isoprene"]

ds = xr.open_dataset(data_file)

var_sequences = []
for var in var_name:
    var_seq = ds[var][0: 6].values
    var_sequences.append(var_seq)

inputs = np.stack(var_sequences, axis=1) # [time=6, channels, lat, lon]
inputs = np.nan_to_num(inputs, nan=0.0)

# 加载 scaler
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

# 归一化
original_shape = inputs.shape  # [time, channels, lat, lon]
inputs_reshaped = inputs.transpose(1, 0, 2, 3).reshape(original_shape[1], -1).T
inputs_normalized = scaler.transform(inputs_reshaped)
inputs_normalized = inputs_normalized.T.reshape(original_shape[1], original_shape[0],
                                               original_shape[2], original_shape[3])
inputs_normalized = inputs_normalized.transpose(1, 0, 2, 3)

# 转成 tensor
inputs = torch.tensor(inputs_normalized, dtype=torch.float32).unsqueeze(0).to(device)
preds = []
print(inputs.shape)
days = 10
for _ in range(days):
    with torch.no_grad():
        pred = model(inputs[:, -6:])  # 用最近6天预测下一天
    preds.append(pred)

    # 把预测结果接到输入末尾
    inputs = torch.cat([inputs, pred], dim=1)

# output =  torch.cat([inputs, preds], dim=1)  # [B, days, C, H, W]

print(inputs.shape)

output = inputs.squeeze(0).cpu().numpy()  # 去掉 batch 维度，变成 [time, channels, lat, lon]

# 构造坐标
time = np.arange(output.shape[0])
channel = var_name
lat = ds['lat'].values if 'lat' in ds else np.arange(output.shape[2])
lon = ds['lon'].values if 'lon' in ds else np.arange(output.shape[3])

# 构造 xarray DataArray
da = xr.DataArray(
    output,
    dims=["time", "channel", "lat", "lon"],
    coords={"time": time, "channel": channel, "lat": lat, "lon": lon},
    name="prediction"
)

# 保存为 netCDF 文件
da.to_netcdf("quanvlstm_prediction.nc")
print("保存完成: quanvlstm_prediction.nc")