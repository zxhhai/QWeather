import xarray as xr
import torch
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, file_path, input_seq_len=8, target_seq_len=1, var_name=['CHLA', 'PAR', 'SST', 'sla', 'tco', 'HCHO', 'windspeed', 'isoprene']):
        self.ds = xr.open_dataset(file_path)
        self.var_name = var_name
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.total_seq_len = input_seq_len + target_seq_len
        self.max_idx = self.ds.sizes['time'] - self.total_seq_len + 1 # 假设时间维度叫day
    
    def __len__(self):
        return self.max_idx
    
    def __getitem__(self, idx):
        if isinstance(self.var_name, list):
            var_sequences = []
            for var in self.var_name:
                var_seq = self.ds[var][idx:idx+self.total_seq_len].values
                var_sequences.append(var_seq)
            seq = torch.stack([torch.tensor(var_seq, dtype=torch.float32) for var_seq in var_sequences], dim=1)
        else:
            seq = self.ds[self.var_name][idx:idx+self.total_seq_len].values
            seq = torch.tensor(seq, dtype=torch.float32)
        input_seq = seq[:self.input_seq_len]
        target_seq = seq[self.input_seq_len:]
        return input_seq, target_seq

def test_weatherdataset():
    from torch.utils.data import DataLoader
    
    dataset = WeatherDataset('/home/zxh/CQ/QWeather/testdata/test_data.nc', input_seq_len=8, target_seq_len=1, var_name='data')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    i = 0
    for input_seq, target_seq in dataloader:
        print(i)
        print("Input sequence shape:", input_seq.shape)
        print("Target sequence shape:", target_seq.shape)
        i += 1

if __name__ == "__main__":
    test_weatherdataset()