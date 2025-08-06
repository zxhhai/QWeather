import torch
import torch.nn as nn
from models.quanvlstm.quanvlstm_model import QuanvLSTMModel
from datasets.dataset import create_dataloaders_with_normalization
from utils.trainer import Trainer
from utils.optimizer import get_optimizer, get_scheduler
from configs.base_config import TrainingConfig


def main():
    config = TrainingConfig(config_path='configs/quanvlstm_config.yaml')

    model = QuanvLSTMModel(
        input_dim=config.model.input_dim,
        hidden_dim_list=config.model.hidden_dim_list,
        kernel_size_list=config.model.kernel_size_list,
        n_qubits_list=config.model.n_qubits_list,
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

    dataloaders = create_dataloaders_with_normalization(
        data_file=config.data.file_path,
        indices_dir=config.data.indices_dir,
        var_name=config.data.var_name,
        input_seq_len=config.data.input_seq_len,
        target_seq_len=config.data.target_seq_len,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        scaler_save_path=config.data.scaler_save_path,
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    trainer = Trainer(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        scheduler=scheduler,
        save_path=config.training.checkpoint_path,
        log_dir=config.logging.log_dir,
        save_logs=config.logging.save_logs
    )

    # Load the latest checkpoint
    if config.training.resume:
        trainer.load_checkpoint(config.training.checkpoint_path + '/latest_checkpoint.pth')

    # training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        save_every=config.training.save_every,
    )

if __name__ == "__main__":
    main()