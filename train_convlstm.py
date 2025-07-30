from models.convlstm.convlstm_model import ConvLSTMModel
from datasets.dataset import WeatherDataset
from utils.trainer import Trainer
from utils.optimizer import get_optimizer, get_scheduler
from configs.base_config import TrainingConfig
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


config = TrainingConfig(config_path='configs/convlstm_config.yaml')

# 确保kernel_size_list是元组列表
kernel_size_list = [tuple(ks) for ks in config.model.kernel_size_list]

model = ConvLSTMModel(
    input_dim=config.model.input_dim,
    hidden_dim_list=config.model.hidden_dim_list,
    kernel_size_list=kernel_size_list,  # 使用转换后的元组列表
    num_layers=config.model.num_layers,
    output_dim=config.model.output_dim,
    T_out=config.model.T_out
)

optimizer = get_optimizer(
    model, 
    config.optimizer.name, 
    lr=config.training.learning_rate
)
scheduler = get_scheduler(
    optimizer, 
    config.scheduler.name, 
    step_size=config.scheduler.step_size, 
    gamma=config.scheduler.gamma
)

trainer = Trainer(
    model=model,
    criterion=nn.MSELoss(),
    optimizer=optimizer,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    scheduler=scheduler,
    save_path='./checkpoints',
)

train_dataset = WeatherDataset(
    file_path=config.data.file_path,
    input_seq_len=config.data.input_seq_len,
    target_seq_len=config.data.target_seq_len,
    var_name=config.data.var_name,
    indices_dir=config.data.indices_dir,
    split='train'
)
val_dataset = WeatherDataset(
    file_path=config.data.file_path,
    input_seq_len=config.data.input_seq_len,
    target_seq_len=config.data.target_seq_len,
    var_name=config.data.var_name,
    indices_dir=config.data.indices_dir,
    split='val'
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=config.training.batch_size, 
    shuffle=True, 
    num_workers=config.data.num_workers
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=config.training.batch_size, 
    shuffle=False, 
    num_workers=config.data.num_workers
)

# training
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=config.training.epochs,
)
