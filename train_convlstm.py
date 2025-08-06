import torch
import torch.nn as nn
from models.convlstm.convlstm_model import ConvLSTMModel
from datasets.dataset import create_dataloaders_with_normalization
from utils.trainer import Trainer
from utils.optimizer import get_optimizer, get_scheduler
from configs.base_config import TrainingConfig
from visualizations.plot_training_history import plot_training_history


def main():
    config = TrainingConfig(config_path='configs/convlstm_config.yaml')

    model = ConvLSTMModel(
        input_dim=config.model.input_dim,
        hidden_dim_list=config.model.hidden_dim_list,
        kernel_size_list=config.model.kernel_size_list,
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

    trainer.save_training_history(save_path=config.training.checkpoint_path + '/training_history.json')
    plot_training_history(trainer.get_training_history())

if __name__ == "__main__":
    main()