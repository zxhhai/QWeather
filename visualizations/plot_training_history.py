import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_training_history(history=None, save_path=None):
    if save_path is not None:
        with open(save_path, 'r') as f:
            history = json.load(f)
    plt.rcParams.update({'font.size': 16})
    sns.set(style="whitegrid", font_scale=1.2)
    epochs = list(range(1, history['current_epoch'] + 1))

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['train_losses'], label='Train Loss', color='royalblue', linewidth=2)
    plt.plot(epochs, history['val_losses'], label='Val Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    if history['metrics_history']:
        metrics_dict = {}
        for record in history['metrics_history']:
            for k, v in record.items():
                metrics_dict.setdefault(k, []).append(v)

        plt.figure(figsize=(12, 8))
        for metric_name, values in metrics_dict.items():
            plt.plot(epochs, values, label=metric_name, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Validation Metrics Over Epochs')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_training_history(save_path='pretrained/quanvlstm_training_history.json')