# 基于量子技术的气象预测

## 一、作品简介
本项目旨在构建机器学习模型ConvLSTM，量子经典混合模型QuanvLSTM，实现对包含异戊二烯等的多维时空气象数据的联合预测。

## 二、环境配置
本项目基于 **Python 3.11** , **torch** 和 **torchquantum** ，建议使用 Conda 创建独立环境。
```bash
conda create -n qweather python=3.11
conda activate qweather
```
### 安装依赖
```bash
pip install -r requirements.txt
```
### 单独安装 torchquantum
```bash
git clone https://github.com/mit-han-lab/torchquantum.git
cd torchquantum
pip install --editable .
cd ..
```

## 三、项目结构
```bash
.
├── README.md
├── checkpoints     # 模型检查点
│   ├── convlstm
│   └── quanvlstm
├── configs         # 配置文件
├── data_preprocess # 数据预处理
├── datasets        # 数据集划分、加载
│   ├── data
│   ├── dataset.py
│   ├── scaler
│   └── split       # 数据集的划分（使用索引）
│       ├── large
│       └── small
├── logs            # 训练日志
│   ├── convlstm
│   └── quanvlstm
├── models          # 模型架构
│   ├── convlstm
│   └── quanvlstm
├── pretrained      # 预训练模型
├── requirements.txt
├── train_convlstm.py
├── train_quanvlstm.py
├── evaluate_convlstm.py
├── evaluate_quanvlstm.py
├── utils           # 训练代码、工具函数
└── visualizations  # 可视化代码
```

## 四、数据准备
原始数据存放于datasets/data，依分辨率的不同分为data_large.nc/data_small.nc，默认使用data_large.nc。

训练集、验证集和测试集的划分位于datasets/split。

## 五、模型训练
### 经典模型 - ConvLSTM
配置好configs/下的convlstm.yaml文件，进入项目根目录，然后运行
```bash
python train_convlstm.py
```
将启动训练
### 量子经典合模型 - QuanvLSTM
配置好configs/下的quanvlstm.yaml文件，进入项目根目录，然后运行
```bash
python train_quanvlstm.py
```
pretrained/文件夹下提供预训练模型convlstm.pth，quanvlstm。

## 六、模型评估
模型的评估在测试集上进行，以平均mse损失为度量指标。

更改evaluate_convlstm.py/evaluate_quanvlstm.py中的model_path为待评估模型保存路径，
默认评估预训练模型pretrained/convlstm.pth以及pretrained/quanvlstm.pth。

进入项目根目录，分别运行
```bash
python evaluate_convlstm.py
python evaluate_quanvlstm.py
```