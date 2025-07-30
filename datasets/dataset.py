import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
import os

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
            if line:  # 跳过空行
                indices.append(int(line))
    
    return np.array(indices)

class WeatherDataset(Dataset):
    def __init__(self, file_path, input_seq_len=8, target_seq_len=1, 
                 var_name=['CHLA', 'PAR', 'SST', 'sla', 'tco', 'HCHO', 'windspeed', 'isoprene'],
                 indices=None, indices_file=None, indices_dir=None, split=None):
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
        """
        self.ds = xr.open_dataset(file_path)
        self.var_name = var_name
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.total_seq_len = input_seq_len + target_seq_len
        
        # 确定要使用的索引
        if indices is not None:
            # 直接使用传入的索引
            self.indices = np.array(indices)
            source_info = f"provided indices: {len(self.indices)} sequences"
        elif indices_file is not None:
            # 从文件加载索引
            self.indices = load_indices_from_txt(indices_file)
            source_info = f"{indices_file}: {len(self.indices)} sequences"
        elif indices_dir is not None and split is not None:
            # 从目录和split加载索引
            indices_file = os.path.join(indices_dir, f'{split}_indices.txt')
            self.indices = load_indices_from_txt(indices_file)
            source_info = f"{split} split from {indices_dir}: {len(self.indices)} sequences"
        else:
            # 使用全部数据
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
        
        print(f"WeatherDataset initialized with {source_info}")
        print(f"  Data shape: {dict(self.ds.sizes)}")
        print(f"  Variables: {self.var_name}")
        print(f"  Sequence length: {self.input_seq_len} + {self.target_seq_len}")
        if len(self.indices) > 0:
            print(f"  Index range: {self.indices.min()} - {self.indices.max()}")
    
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
            seq = torch.stack([torch.tensor(var_seq, dtype=torch.float32) for var_seq in var_sequences], dim=1)
        else:
            seq = self.ds[self.var_name][actual_idx:actual_idx+self.total_seq_len].values
            seq = torch.tensor(seq, dtype=torch.float32)
            # 为单变量添加channel维度
            if seq.ndim == 3:  # [time, lat, lon]
                seq = seq.unsqueeze(1)  # [time, 1, lat, lon]
        
        input_seq = seq[:self.input_seq_len]
        target_seq = seq[self.input_seq_len:]

        # 用0代替nan值
        input_seq = torch.nan_to_num(input_seq, nan=0.0)
        target_seq = torch.nan_to_num(target_seq, nan=0.0)
        return input_seq, target_seq
    
    def get_info(self):
        """获取数据集信息"""
        return {
            'dataset_size': len(self),
            'total_sequences_in_file': self.ds.sizes['time'] - self.total_seq_len + 1,
            'input_seq_len': self.input_seq_len,
            'target_seq_len': self.target_seq_len,
            'variables': self.var_name,
            'spatial_shape': (self.ds.dims.get('lat', 'N/A'), self.ds.dims.get('lon', 'N/A')),
            'index_range': (int(self.indices.min()), int(self.indices.max())) if len(self.indices) > 0 else None
        }
    
    def close(self):
        """关闭数据集"""
        if hasattr(self, 'ds'):
            self.ds.close()

def create_dataloaders_from_indices(data_file, indices_dir, var_name, 
                                  input_seq_len=8, target_seq_len=1,
                                  batch_size=32, num_workers=4):
    """
    根据索引目录创建DataLoader
    
    Args:
        data_file: 原始数据文件
        indices_dir: 索引文件目录
        var_name: 变量名
        input_seq_len: 输入序列长度
        target_seq_len: 目标序列长度
        batch_size: 批量大小
        num_workers: 工作进程数
    
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    from torch.utils.data import DataLoader
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        indices_file = os.path.join(indices_dir, f'{split}_indices.txt')
        
        if os.path.exists(indices_file):
            dataset = WeatherDataset(
                file_path=data_file,
                input_seq_len=input_seq_len,
                target_seq_len=target_seq_len,
                var_name=var_name,
                indices_dir=indices_dir,
                split=split
            )
            
            shuffle = (split == 'train')
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            dataloaders[split] = dataloader
            print(f"Created {split} DataLoader: {len(dataset)} samples, {len(dataloader)} batches")
        else:
            print(f"Warning: {indices_file} not found, skipping {split}")
    
    return dataloaders

def test_weatherdataset():
    """测试WeatherDataset的各种用法"""
    print("="*60)
    print("Testing WeatherDataset")
    print("="*60)
    
    data_file = '/home/zxh/CQ/dataset/isoprene_results.nc'
    var_names = ['CHLA', 'PAR', 'SST', 'sla', 'tco', 'HCHO', 'windspeed', 'isoprene']
    
    # 测试1: 完整数据集
    print("\n1. Testing full dataset...")
    dataset_full = WeatherDataset(data_file, var_name=['CHLA', 'PAR', 'SST', 'sla', 'tco', 'HCHO', 'windspeed', 'isoprene'])
    print(f"Full dataset info: {dataset_full.get_info()}")
    
    # 测试2: 使用索引目录
    indices_dir = "data/indices"
    if os.path.exists(indices_dir):
        print("\n2. Testing with indices directory...")
        
        for split in ['train', 'val', 'test']:
            indices_file = os.path.join(indices_dir, f'{split}_indices.txt')
            if os.path.exists(indices_file):
                dataset_split = WeatherDataset(
                    data_file, 
                    var_name=var_names,
                    indices_dir=indices_dir,
                    split=split
                )
                print(f"{split} dataset info: {dataset_split.get_info()}")
    
    # 测试3: DataLoader
    if os.path.exists(indices_dir):
        print("\n3. Testing DataLoader creation...")
        dataloaders = create_dataloaders_from_indices(
            data_file=data_file,
            indices_dir=indices_dir,
            var_name=['CHLA', 'PAR', 'SST', 'sla', 'tco', 'HCHO', 'windspeed', 'isoprene'],
            batch_size=4,
            num_workers=2
        )
        
        for split_name, dataloader in dataloaders.items():
            print(f"\n{split_name} DataLoader test:")
            for i, (input_seq, target_seq) in enumerate(dataloader):
                print(f"  Batch {i}: input={input_seq.shape}, target={target_seq.shape}")
                if i >= 0:  # 只测试第一个batch
                    break

if __name__ == "__main__":
    test_weatherdataset()