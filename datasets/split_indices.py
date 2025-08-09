import xarray as xr
import numpy as np
import os
import json

def create_split_indices(data_file, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, 
                        input_seq_len=8, target_seq_len=1, seed=42):
    """
    创建数据集划分索引并保存到txt文件
    
    Args:
        data_file: 原始数据文件路径
        output_dir: 索引文件保存目录
        train_ratio, val_ratio, test_ratio: 划分比例
        input_seq_len, target_seq_len: 序列长度
        seed: 随机种子
    
    Returns:
        dict: 包含各split索引的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据获取时间维度大小
    print(f"Reading data file: {data_file}")
    ds = xr.open_dataset(data_file)
    total_seq_len = input_seq_len + target_seq_len
    total_sequences = ds.sizes['time'] - total_seq_len + 1
    ds.close()
    
    print(f"Total available sequences: {total_sequences}")
    
    # 生成所有索引并随机打乱
    all_indices = np.arange(total_sequences)
    #np.random.seed(seed)
    #np.random.shuffle(all_indices)
    
    # 计算划分点
    train_end = int(total_sequences * train_ratio)
    val_end = int(total_sequences * (train_ratio + val_ratio))
    
    # 划分索引
    splits = {
        'train': all_indices[:train_end],
        'val': all_indices[train_end:val_end],
        'test': all_indices[val_end:]
    }
    
    # 保存索引到txt文件
    for split_name, indices in splits.items():
        # 保存为txt格式（每行一个索引）
        txt_file = os.path.join(output_dir, f'{split_name}_indices.txt')
        with open(txt_file, 'w') as f:
            for idx in indices:
                f.write(f"{idx}\n")
        
        print(f"Saved {split_name} indices: {len(indices)} sequences to {txt_file}")
        print(f"  Index range: {indices[0]} - {indices[-1]}")
        print(f"  Sample indices: {indices[:5].tolist()}")
    
    # 保存元数据
    metadata = {
        'data_file': data_file,
        'total_sequences': int(total_sequences),
        'input_seq_len': input_seq_len,
        'target_seq_len': target_seq_len,
        'total_seq_len': total_seq_len,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'seed': seed,
        'splits': {k: len(v) for k, v in splits.items()},
        'train_end': int(train_end),
        'val_end': int(val_end)
    }
    
    metadata_file = os.path.join(output_dir, 'split_info.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {metadata_file}")
    
    # 打印划分统计信息
    print(f"\nSplit Statistics:")
    print(f"  Train: {len(splits['train'])} sequences ({len(splits['train'])/total_sequences:.1%})")
    print(f"  Val:   {len(splits['val'])} sequences ({len(splits['val'])/total_sequences:.1%})")
    print(f"  Test:  {len(splits['test'])} sequences ({len(splits['test'])/total_sequences:.1%})")
    
    return splits

def load_split_indices(indices_dir, split='train'):
    """
    从txt文件加载划分索引
    
    Args:
        indices_dir: 索引文件目录
        split: 'train', 'val', 'test'
    
    Returns:
        numpy.ndarray: 索引数组
    """
    txt_file = os.path.join(indices_dir, f'{split}_indices.txt')
    
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Index file not found: {txt_file}")
    
    indices = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                indices.append(int(line))
    
    return np.array(indices)

def load_split_metadata(indices_dir):
    """
    加载划分元数据
    
    Args:
        indices_dir: 索引文件目录
    
    Returns:
        dict: 元数据字典
    """
    metadata_file = os.path.join(indices_dir, 'split_info.json')
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def validate_split_indices(indices_dir):
    """
    验证索引文件的完整性和正确性
    
    Args:
        indices_dir: 索引文件目录
    
    Returns:
        bool: 验证是否通过
    """
    try:
        # 加载元数据
        metadata = load_split_metadata(indices_dir)
        print(f"Validating split indices in: {indices_dir}")
        print(f"Original data file: {metadata['data_file']}")
        
        # 加载所有split的索引
        all_loaded_indices = []
        for split in ['train', 'val', 'test']:
            indices = load_split_indices(indices_dir, split)
            all_loaded_indices.extend(indices.tolist())
            print(f"  {split}: {len(indices)} indices")
        
        # 检查索引范围
        all_loaded_indices = np.array(all_loaded_indices)
        expected_total = metadata['total_sequences']
        
        print(f"\nValidation Results:")
        print(f"  Expected total sequences: {expected_total}")
        print(f"  Loaded indices count: {len(all_loaded_indices)}")
        print(f"  Index range: {all_loaded_indices.min()} - {all_loaded_indices.max()}")
        print(f"  Unique indices: {len(np.unique(all_loaded_indices))}")
        print(f"  Duplicates: {len(all_loaded_indices) - len(np.unique(all_loaded_indices))}")
        
        # 验证条件
        is_valid = True
        if len(np.unique(all_loaded_indices)) != len(all_loaded_indices):
            print("  ❌ Error: Duplicate indices found")
            is_valid = False
        
        if all_loaded_indices.min() < 0 or all_loaded_indices.max() >= expected_total:
            print("  ❌ Error: Indices out of range")
            is_valid = False
        
        if len(all_loaded_indices) != sum(metadata['splits'].values()):
            print("  ❌ Error: Total indices count mismatch")
            is_valid = False
        
        if is_valid:
            print("  ✅ All validations passed")
        
        return is_valid
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

def show_split_info(indices_dir):
    """
    显示划分信息
    
    Args:
        indices_dir: 索引文件目录
    """
    try:
        metadata = load_split_metadata(indices_dir)
        
        print(f"Split Information:")
        print(f"{'='*50}")
        print(f"Data file: {metadata['data_file']}")
        print(f"Total sequences: {metadata['total_sequences']}")
        print(f"Sequence length: {metadata['input_seq_len']} + {metadata['target_seq_len']} = {metadata['total_seq_len']}")
        print(f"Random seed: {metadata['seed']}")
        print(f"Split ratios: {metadata['train_ratio']:.1f} / {metadata['val_ratio']:.1f} / {metadata['test_ratio']:.1f}")
        
        print(f"\nSplit Details:")
        for split, count in metadata['splits'].items():
            percentage = count / metadata['total_sequences'] * 100
            print(f"  {split.capitalize()}: {count:,} sequences ({percentage:.1f}%)")
        
        # 显示每个split的详细信息
        print(f"\nIndex Files:")
        for split in ['train', 'val', 'test']:
            txt_file = os.path.join(indices_dir, f'{split}_indices.txt')
            if os.path.exists(txt_file):
                indices = load_split_indices(indices_dir, split)
                print(f"  {txt_file}")
                print(f"    Count: {len(indices)}")
                print(f"    Range: {indices.min()} - {indices.max()}")
                print(f"    Sample: {indices[:5].tolist()}")
            else:
                print(f"  {txt_file} - NOT FOUND")
                
    except Exception as e:
        print(f"Error loading split info: {e}")

if __name__ == "__main__":
    # 示例用法
    data_file = "/home/zxh/CQ/dataset/all_dataset.nc"
    output_dir = "split/large"
    
    print("Creating split indices...")
    splits = create_split_indices(
        data_file=data_file,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        input_seq_len=6,
        target_seq_len=1,
        seed=42
    )
    
    print("\nValidating indices...")
    validate_split_indices(output_dir)
    
    print("\nShowing split info...")
    show_split_info(output_dir)