import xarray as xr
import numpy as np
import os
from pathlib import Path
import gc

def split_dataset_to_files(input_file, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, 
                          input_seq_len=8, target_seq_len=1, seed=42):
    """
    将数据集随机划分并保存到不同文件
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取数据
    print("Loading dataset...")
    ds = xr.open_dataset(input_file)
    
    # 计算可用的时间序列切片
    total_seq_len = input_seq_len + target_seq_len
    total_sequences = ds.sizes['time'] - total_seq_len + 1
    
    print(f"Total available sequences: {total_sequences}")
    
    # 生成所有可能的起始索引
    all_indices = np.arange(total_sequences)
    
    # 随机打乱
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    
    # 计算划分点
    train_end = int(total_sequences * train_ratio)
    val_end = int(total_sequences * (train_ratio + val_ratio))
    
    # 划分索引
    splits = {
        'train': all_indices[:train_end],
        'val': all_indices[train_end:val_end],
        'test': all_indices[val_end:]
    }
    
    for split_name, indices in splits.items():
        print(f"\nCreating {split_name} dataset with {len(indices)} sequences...")
        
        # 分批处理，但直接写入文件而不是合并
        batch_size = 20  # 减小batch size
        n_batches = (len(indices) + batch_size - 1) // batch_size
        
        output_file = os.path.join(output_dir, f'{split_name}.nc')
        
        # 第一个batch - 创建文件
        first_batch = True
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            print(f"  Processing batch {batch_idx + 1}/{n_batches} ({len(batch_indices)} sequences)...")
            
            # 为当前batch创建时间序列数据
            batch_data = {}
            
            # 对每个变量分别处理
            for var_name in ds.data_vars:
                var_sequences = []
                for seq_idx in batch_indices:
                    time_slice = slice(seq_idx, seq_idx + total_seq_len)
                    seq_data = ds[var_name][time_slice].values
                    var_sequences.append(seq_data)
                
                # 堆叠当前变量的所有序列
                batch_data[var_name] = np.stack(var_sequences, axis=0)
            
            # 创建batch的数据集
            batch_ds = xr.Dataset({
                var_name: (['sequence', 'time_in_seq', 'lat', 'lon'], data)
                for var_name, data in batch_data.items()
            }, coords={
                'sequence': np.arange(start_idx, start_idx + len(batch_indices)),
                'time_in_seq': np.arange(total_seq_len),
                'lat': ds.lat,
                'lon': ds.lon,
            })
            
            if first_batch:
                # 第一个batch：创建文件
                batch_ds.attrs['input_seq_len'] = input_seq_len
                batch_ds.attrs['target_seq_len'] = target_seq_len
                batch_ds.attrs['total_seq_len'] = total_seq_len
                batch_ds.attrs['original_file'] = input_file
                batch_ds.attrs['split'] = split_name
                batch_ds.attrs['random_seed'] = seed
                
                batch_ds.to_netcdf(output_file)
                first_batch = False
            else:
                # 后续batch：追加到文件
                # 重新读取现有文件
                existing_ds = xr.open_dataset(output_file)
                
                # 合并当前batch
                combined_ds = xr.concat([existing_ds, batch_ds], dim='sequence')
                
                # 重新编号sequence
                combined_ds = combined_ds.assign_coords(sequence=np.arange(len(combined_ds.sequence)))
                
                # 保持属性
                combined_ds.attrs = existing_ds.attrs
                
                # 保存合并后的数据
                existing_ds.close()
                combined_ds.to_netcdf(output_file)
                combined_ds.close()
                
                del existing_ds, combined_ds
            
            # 清理当前batch的内存
            batch_ds.close()
            del batch_data, batch_ds
            gc.collect()
        
        print(f"  {split_name} dataset saved to {output_file}")
    
    ds.close()
    print("\nDataset splitting completed!")

if __name__ == "__main__":
    # 执行数据集分割
    input_file = "/home/zxh/CQ/dataset/isoprene_results.nc"
    output_dir = "data/large"
    
    split_dataset_to_files(
        input_file=input_file,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        input_seq_len=8,
        target_seq_len=1,
        seed=42
    )

