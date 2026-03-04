import copy
import os
from pathlib import Path
from typing import Optional

import yaml


class ConfigError(Exception):
    pass


def _wrap(value):
    """Wrap dict values in _ConfigNode; return everything else as-is."""
    return _ConfigNode(value) if isinstance(value, dict) else value


class _ConfigNode:
    """Lazy dict wrapper providing attribute and item access on config sub-sections."""

    def __init__(self, data: dict):
        object.__setattr__(self, '_data', data)

    def __getattr__(self, key):
        data = object.__getattribute__(self, '_data')
        if key not in data:
            raise AttributeError(f"Config has no key '{key}'")
        return _wrap(data[key])

    def __getitem__(self, key):
        return _wrap(object.__getattribute__(self, '_data')[key])

    def __repr__(self):
        return f"_ConfigNode({object.__getattribute__(self, '_data')!r})"


class Config:
    """Top-level YAML configuration loader.

    Resolution order for config file path:
        1. ``path`` constructor argument
        2. ``CONFIG_PATH`` environment variable
        3. ``./config.yaml`` in the current working directory

    Usage::

        config = Config()
        config.training.learning_rate          # attribute access
        config['training']['learning_rate']    # item access
        config.get('training.learning_rate', 1e-4)  # dotted key with default
        config.merge({'training': {'epochs': 100}}) # deep-merge overrides
    """

    def __init__(self, path: Optional[str] = None):
        if path is not None:
            config_path = Path(path)
        elif 'CONFIG_PATH' in os.environ:
            config_path = Path(os.environ['CONFIG_PATH'])
        else:
            config_path = Path('config.yaml')

        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse config file {config_path}: {e}")

        if not isinstance(data, dict):
            raise ConfigError(f"Config file must contain a YAML mapping at the top level: {config_path}")

        self._data = data

    def __getattr__(self, key):
        if key.startswith('_'):
            raise AttributeError(key)
        data = object.__getattribute__(self, '_data')
        if key not in data:
            raise AttributeError(f"Config has no section '{key}'")
        return _wrap(data[key])

    def __getitem__(self, key):
        return _wrap(self._data[key])

    def get(self, dotted_key: str, default=None):
        """Retrieve a value by dotted key, returning ``default`` on any missing key.

        Example::

            config.get('training.learning_rate', 1e-4)
        """
        keys = dotted_key.split('.')
        node = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def merge(self, overrides: dict):
        """Deep-merge ``overrides`` into the config, skipping ``None`` values."""
        self._deep_merge(self._data, overrides)

    def _deep_merge(self, base: dict, overrides: dict):
        for key, value in overrides.items():
            if value is None:
                continue
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def to_dict(self) -> dict:
        """Return a deep copy of the raw config dictionary."""
        return copy.deepcopy(self._data)

    def __repr__(self):
        return f"Config({self._data!r})"
