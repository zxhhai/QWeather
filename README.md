# 基于量子技术的气象预测

## 一、作品简介

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



## 五、模型训练
### 经典模型 - ConvLSTM
配置好configs/下的convlstm.yaml文件，进入项目根目录，然后运行
```bash
python train_convlstm.py
```
将启动训练
### 经典量子混合模型 - QuanvLSTM
配置好configs/下的quanvlstm.yaml文件，进入项目根目录，然后运行
```bash
python train_quanvlstm.py
```

## 六、模型评估
更改