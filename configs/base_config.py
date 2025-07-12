import yaml
import os
from typing import Dict, Any


class BaseConfig:
    """Base configuration class"""
    
    def __init__(self, config_path: str = None, **kwargs):
        """
        Initialize configuration
        
        Args:
            config_path: YAML configuration file path
            **kwargs: Direct configuration parameters
        """
        self.config = {}
        
        # Load configuration from YAML file
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
        
        # Update configuration
        self.config.update(kwargs)
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def save_to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.config.copy()


class TrainingConfig(BaseConfig):
    """Training configuration class"""
    
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)
        
        # Set default values
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration"""
        defaults = {
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'early_stopping_patience': 20,
                'save_every': 5
            },
            'optimizer': {
                'name': 'adam',
                'momentum': 0.9
            },
            'scheduler': {
                'name': 'step',
                'step_size': 30,
                'gamma': 0.1,
                'T_max': 200,
                'patience': 10
            },
            'model': {
                'name': 'lstm',
                'input_size': 10,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2
            },
            'data': {
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1,
                'shuffle': True
            }
        }
        
        # Only set default values that don't exist
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    @property
    def epochs(self):
        return self.get('training.epochs')
    
    @property
    def batch_size(self):
        return self.get('training.batch_size')
    
    @property
    def learning_rate(self):
        return self.get('training.learning_rate')
    
    @property
    def optimizer_name(self):
        return self.get('optimizer.name')
    
    @property
    def scheduler_name(self):
        return self.get('scheduler.name')


def load_config(config_path: str) -> BaseConfig:
    """Convenient function to load configuration file"""
    return BaseConfig(config_path)


def create_default_configs():
    """Create default configuration files"""
    config_dir = os.path.dirname(__file__)
    
    # Training configuration
    training_config = {
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'early_stopping_patience': 10,
            'save_every': 5
        },
        'optimizer': {
            'name': 'adam',
            'momentum': 0.9
        },
        'scheduler': {
            'name': 'step',
            'step_size': 30,
            'gamma': 0.1
        }
    }
    
    config = BaseConfig()
    config.config = training_config
    config.save_to_yaml(os.path.join(config_dir, 'default_training.yaml'))
    
    print("Default configuration files created!")


if __name__ == "__main__":
    create_default_configs()
