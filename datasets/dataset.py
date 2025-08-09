import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_indices_from_txt(indices_file):
    """
    从txt文件加载索引
    
    Args:
        indices_file: txt索引文件路径
    
    Returns:
        numpy.ndarray: 索引数组
    """
    if not os.path.exists(indices_file):
        raise FileNotFoundError(f"Index file not found: {indices_file}")
    
    indices = []
    with open(indices_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                indices.append(int(line))
    
    return np.array(indices)

class WeatherDataset(Dataset):
    def __init__(self, file_path, input_seq_len=8, target_seq_len=1, 
                 var_name=['CHLA', 'PAR', 'SST', 'sla', 'tco', 'HCHO', 'windspeed', 'isoprene'],
                 indices=None, indices_file=None, indices_dir=None, split=None,
                 normalize=True, scaler_type='standard', scaler=None, fit_scaler=False):
        """
        Args:
            file_path: 原始数据文件路径
            input_seq_len: 输入序列长度
            target_seq_len: 目标序列长度
            var_name: 变量名列表
            indices: 直接传入的索引数组
            indices_file: 索引文件路径（.txt）
            indices_dir: 索引文件目录（配合split使用）
            split: 'train', 'val', 'test'（配合indices_dir使用）
            normalize: 是否进行归一化
            scaler_type: 'standard' 或 'minmax'
            scaler: 预训练的scaler对象（用于val/test）
            fit_scaler: 是否训练scaler（只在train时设为True）
        """
        self.ds = xr.open_dataset(file_path)
        self.var_name = var_name
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.total_seq_len = input_seq_len + target_seq_len
        self.normalize = normalize

        self.scaler = scaler
        
        # 确定要使用的索引
        if indices is not None:
            self.indices = np.array(indices)
            source_info = f"provided indices: {len(self.indices)} sequences"
        elif indices_file is not None:
            self.indices = load_indices_from_txt(indices_file)
            source_info = f"{indices_file}: {len(self.indices)} sequences"
        elif indices_dir is not None and split is not None:
            indices_file = os.path.join(indices_dir, f'{split}_indices.txt')
            self.indices = load_indices_from_txt(indices_file)
            source_info = f"{split} split from {indices_dir}: {len(self.indices)} sequences"
        else:
            total_sequences = self.ds.sizes['time'] - self.total_seq_len + 1
            self.indices = np.arange(total_sequences)
            source_info = f"full dataset: {len(self.indices)} sequences"
        
        # 验证索引范围
        max_valid_idx = self.ds.sizes['time'] - self.total_seq_len
        if len(self.indices) > 0:
            if self.indices.max() > max_valid_idx:
                raise ValueError(f"Index {self.indices.max()} exceeds maximum valid index {max_valid_idx}")
            if self.indices.min() < 0:
                raise ValueError(f"Negative index found: {self.indices.min()}")
        
        # 训练scaler（只在训练集）
        if self.normalize and fit_scaler:
            self._fit_scaler(scaler_type)
        elif self.normalize and self.scaler is None:
            print("Warning: normalize=True but no scaler provided")
            self.normalize = False
        
        print(f"WeatherDataset initialized with {source_info}")

    
    def _fit_scaler(self, scaler_type='standard', sample_size=100):
        """训练scaler"""
        print(f"Fitting {scaler_type} scaler...")
        
        # 创建scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # 收集训练数据
        sample_indices = np.random.choice(
            self.indices, 
            min(sample_size, len(self.indices)), 
            replace=False
        )
        
        all_data = []
        for idx in sample_indices[:50]:  # 最多50个样本
            if isinstance(self.var_name, list):
                var_sequences = []
                for var in self.var_name:
                    var_seq = self.ds[var][idx:idx+self.total_seq_len].values
                    var_sequences.append(var_seq)
                seq = np.stack(var_sequences, axis=1)  # [time, channels, lat, lon]
            else:
                seq = self.ds[self.var_name][idx:idx+self.total_seq_len].values
                seq = seq[:, np.newaxis, :, :]
            
            # 将数据reshape为2D：[time*lat*lon, channels]
            seq = np.nan_to_num(seq, nan=0.0)
            seq_reshaped = seq.transpose(1, 0, 2, 3).reshape(seq.shape[1], -1).T
            all_data.append(seq_reshaped)
        
        # 合并所有数据并训练scaler
        all_data = np.concatenate(all_data, axis=0)  # [samples, channels]
        self.scaler.fit(all_data)
        
        # print(f"Scaler fitted on {all_data.shape[0]} samples")
        # if scaler_type == 'standard':
            # print(f"  Mean: {self.scaler.mean_}")
            # print(f"  Scale: {self.scaler.scale_}")
        # else:
            # print(f"  Min: {self.scaler.data_min_}")
            # print(f"  Max: {self.scaler.data_max_}")
    
    def get_scaler(self):
        return self.scaler
    
    def save_scaler(self, save_path):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if self.scaler:
            with open(save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            #print(f"Scaler saved to {save_path}")
    
    @staticmethod
    def load_scaler(load_path):
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    
    def _normalize_data(self, data):
        if not self.normalize or self.scaler is None:
            return data
        
        # data shape: [time, channels, lat, lon]
        original_shape = data.shape
        
        # 重塑为2D: [time*lat*lon, channels]
        data_reshaped = data.transpose(1, 0, 2, 3).reshape(data.shape[1], -1).T
        
        # 归一化
        data_normalized = self.scaler.transform(data_reshaped)
        
        # 重塑回原始形状
        data_normalized = data_normalized.T.reshape(original_shape[1], original_shape[0], 
                                                  original_shape[2], original_shape[3])
        data_normalized = data_normalized.transpose(1, 0, 2, 3)
        
        return data_normalized
    
    def denormalize_data(self, data):
        if not self.normalize or self.scaler is None:
            return data
        
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            is_tensor = True
            device = data.device
        else:
            data_np = data
            is_tensor = False
        
        # data shape: [batch, time, channels, lat, lon] 或 [time, channels, lat, lon]
        original_shape = data_np.shape
        
        if len(original_shape) == 5:  # batch维度
            # 重塑为2D: [batch*time*lat*lon, channels]
            data_reshaped = data_np.transpose(0, 2, 1, 3, 4).reshape(-1, original_shape[2])
        else:  # 无batch维度
            # 重塑为2D: [time*lat*lon, channels]  
            data_reshaped = data_np.transpose(1, 0, 2, 3).reshape(original_shape[1], -1).T
        
        # 反归一化
        data_denormalized = self.scaler.inverse_transform(data_reshaped)
        
        # 重塑回原始形状
        if len(original_shape) == 5:
            data_denormalized = data_denormalized.reshape(original_shape[0], original_shape[2], 
                                                        original_shape[1], original_shape[3], 
                                                        original_shape[4])
            data_denormalized = data_denormalized.transpose(0, 2, 1, 3, 4)
        else:
            data_denormalized = data_denormalized.T.reshape(original_shape[1], original_shape[0], 
                                                          original_shape[2], original_shape[3])
            data_denormalized = data_denormalized.transpose(1, 0, 2, 3)
        
        # 转换回tensor
        if is_tensor:
            return torch.tensor(data_denormalized, device=device, dtype=torch.float32)
        else:
            return data_denormalized
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.indices)}")
        
        # 获取实际的时间索引
        actual_idx = self.indices[idx]
        
        if isinstance(self.var_name, list):
            var_sequences = []
            for var in self.var_name:
                var_seq = self.ds[var][actual_idx:actual_idx+self.total_seq_len].values
                var_sequences.append(var_seq)
            seq = np.stack(var_sequences, axis=1)  # [time, channels, lat, lon]
        else:
            seq = self.ds[self.var_name][actual_idx:actual_idx+self.total_seq_len].values
            seq = seq[:, np.newaxis, :, :]  # [time, 1, lat, lon]
        
        # 处理NaN值
        seq = np.nan_to_num(seq, nan=0.0)
        
        # 归一化
        if self.normalize:
            seq = self._normalize_data(seq)
        
        # 转换为tensor
        seq = torch.tensor(seq, dtype=torch.float32)
        
        input_seq = seq[:self.input_seq_len]
        target_seq = seq[self.input_seq_len:]
        
        return input_seq, target_seq
    
    def get_info(self):
        """获取数据集信息"""
        info = {
            'dataset_size': len(self),
            'total_sequences_in_file': self.ds.sizes['time'] - self.total_seq_len + 1,
            'input_seq_len': self.input_seq_len,
            'target_seq_len': self.target_seq_len,
            'variables': self.var_name,
            'spatial_shape': (self.ds.dims.get('lat', 'N/A'), self.ds.dims.get('lon', 'N/A')),
            'index_range': (int(self.indices.min()), int(self.indices.max())) if len(self.indices) > 0 else None,
            'normalized': self.normalize
        }
        return info
    
    def close(self):
        """关闭数据集"""
        if hasattr(self, 'ds'):
            self.ds.close()

def create_dataloader(
    data_file, indices_dir, var_name, split,
    input_seq_len=8, target_seq_len=1,
    batch_size=32, num_workers=4,
    normalize=True, scaler_type='standard',
    scaler_file='./datasets/scaler.pkl'
):
    from torch.utils.data import DataLoader

    if split == 'train':
        dataset = WeatherDataset(
            file_path=data_file,
            input_seq_len=input_seq_len,
            target_seq_len=target_seq_len,
            var_name=var_name,
            indices_dir=indices_dir,
            split='train',
            normalize=normalize,
            scaler_type=scaler_type,
            fit_scaler=normalize
        )
        if normalize and scaler_file:
            dataset.save_scaler(scaler_file)
    else:
        scaler = WeatherDataset.load_scaler(scaler_file) if scaler_file else None
        dataset = WeatherDataset(
            file_path=data_file,
            input_seq_len=input_seq_len,
            target_seq_len=target_seq_len,
            var_name=var_name,
            indices_dir=indices_dir,
            split=split,
            normalize=normalize,
            scaler=scaler,
            fit_scaler=None
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return dataloader