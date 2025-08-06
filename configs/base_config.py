import yaml
import os
from typing import Dict, Any


class AttrDict(dict):
    """字典的属性访问版本，支持嵌套"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 递归转换所有嵌套字典为AttrDict
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, AttrDict):
                self[key] = AttrDict(value)
            elif isinstance(value, list):
                self[key] = self._convert_list(value)
    
    def _convert_list(self, lst):
        """递归转换列表中的字典"""
        new_list = []
        for item in lst:
            if isinstance(item, dict) and not isinstance(item, AttrDict):
                new_list.append(AttrDict(item))
            elif isinstance(item, list):
                new_list.append(self._convert_list(item))
            else:
                new_list.append(item)
        return new_list
    
    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
        
        # 确保返回的字典也是AttrDict
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[item] = value
        
        return value

    def __setattr__(self, key, value):
        # 如果设置的是字典，转换为AttrDict
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


class BaseConfig:
    """Base configuration class"""

    def __init__(self, config_path: str = None, **kwargs):
        """
        Initialize configuration

        Args:
            config_path: YAML configuration file path
            **kwargs: Direct configuration parameters
        """
        self.config = AttrDict()

        # Load configuration from YAML file
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)

        # Update configuration
        if kwargs:
            self.config.update(AttrDict(kwargs))

    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # 确保完全转换为AttrDict
        self.config = AttrDict(data) if data else AttrDict()

    def save_to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dict(self.config), f, default_flow_style=False, allow_unicode=True)

    def get(self, key: str, default=None):
        """Get configuration value by dotted key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set configuration value by dotted key"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = AttrDict()
            config = config[k]
        config[keys[-1]] = value

    # 添加直接属性访问支持
    def __getattr__(self, item):
        """支持 config.model 这样的访问方式"""
        if item == 'config':
            return object.__getattribute__(self, 'config')
        try:
            return self.config[item]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        """支持 config.model = {...} 这样的设置方式"""
        if key == 'config':
            object.__setattr__(self, key, value)
        else:
            if not hasattr(self, 'config'):
                object.__setattr__(self, key, value)
            else:
                # 确保设置的字典值也转换为AttrDict
                if isinstance(value, dict) and not isinstance(value, AttrDict):
                    value = AttrDict(value)
                self.config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary"""
        def convert(d):
            if isinstance(d, dict):
                return {k: convert(v) for k, v in d.items()}
            else:
                return d
        return convert(self.config)


class TrainingConfig(BaseConfig):
    """Training configuration class"""

    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)
        self._set_defaults()

    def _set_defaults(self):
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

        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = AttrDict(value)
            else:
                # 递归补全缺失的默认值
                self._fill_defaults(self.config[key], value)
        
        # 转换 kernel_size_list 为 tuple 列表
        if hasattr(self.config.model, 'kernel_size_list'):
            self.config.model.kernel_size_list = [
                tuple(x) for x in self.config.model.kernel_size_list
            ]

    def _fill_defaults(self, current, defaults):
        """递归填充默认值，确保所有字典都是AttrDict"""
        for k, v in defaults.items():
            if k not in current:
                if isinstance(v, dict):
                    current[k] = AttrDict(v)
                else:
                    current[k] = v
            elif isinstance(v, dict) and isinstance(current[k], dict):
                # 确保当前值也是AttrDict
                if not isinstance(current[k], AttrDict):
                    current[k] = AttrDict(current[k])
                self._fill_defaults(current[k], v)

    @property
    def epochs(self):
        return self.config.training.epochs

    @property
    def batch_size(self):
        return self.config.training.batch_size

    @property
    def learning_rate(self):
        return self.config.training.learning_rate

    @property
    def optimizer_name(self):
        return self.config.optimizer.name

    @property
    def scheduler_name(self):
        return self.config.scheduler.name