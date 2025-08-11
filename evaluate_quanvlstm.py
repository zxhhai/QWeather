import torch
from models.quanvlstm.quanvlstm_model import QuanvLSTMModel
from datasets.dataset import create_dataloader
from utils.evaluater import Evaluater


def main():
    test_loader = create_dataloader(
        data_file="./datasets/data/data_large.nc",
        indices_dir="./datasets/split/large",
        var_name=["chla", "hcho", "no2", "o3", "par", "sla", "sst", "wind", "DML", "isoprene"],
        split='test',
        input_seq_len=6,
        target_seq_len=1,
        batch_size=1,
        num_workers=4,
        normalize=True,
        scaler_type="standard",
        scaler_file='./datasets/scaler/quanvlstm_scaler.pkl'
    )

    evaluater = Evaluater(
        model_class=QuanvLSTMModel,
        model_kwargs=dict(
            input_dim=10,
            hidden_dim_list=[32, 16],
            kernel_size_list=[(3, 3), (3, 3)],
            num_layers=2,
            output_dim=10,
            T_out=1,
            n_qubits_list=[4, 4]
        ),
        model_path="./pretrained/quanvlstm.pth", # 待评估模型保存路径
        dataloader=test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    evaluater.evaluate_mse()

if __name__ == "__main__":
    main()