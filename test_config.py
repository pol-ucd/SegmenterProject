"""
Unit tests for Config class in segmenter/core/config.py
"""

import os
import tempfile
import sys
import yaml
from pathlib import Path

from segmenter.core.config import Config, ConfigError


def test_load_config_from_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'training': {'epochs': 100, 'learning_rate': 1e-4}}, f)
        config_path = f.name
    
    try:
        config = Config(path=config_path)
        assert config.get('training.epochs') == 100
        assert config.get('training.learning_rate') == 1e-4
        print("PASS: test_load_config_from_file")
    finally:
        os.unlink(config_path)


def test_load_nonexistent_config_raises_error():
    try:
        Config(path='/nonexistent/config.yaml')
        assert False, "Expected ConfigError"
    except ConfigError as e:
        assert "Config file not found" in str(e)
        print("PASS: test_load_nonexistent_config_raises_error")


def test_invalid_yaml_raises_error():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        config_path = f.name
    
    try:
        try:
            Config(path=config_path)
            assert False, "Expected ConfigError"
        except ConfigError as e:
            assert "Failed to parse" in str(e)
            print("PASS: test_invalid_yaml_raises_error")
    finally:
        os.unlink(config_path)


def test_non_mapping_raises_error():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump([1, 2, 3], f)
        config_path = f.name
    
    try:
        try:
            Config(path=config_path)
            assert False, "Expected ConfigError"
        except ConfigError as e:
            assert "YAML mapping" in str(e)
            print("PASS: test_non_mapping_raises_error")
    finally:
        os.unlink(config_path)


def test_get_with_default():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'training': {'epochs': 50}}, f)
        config_path = f.name
    
    try:
        config = Config(path=config_path)
        assert config.get('training.epochs', 10) == 50
        assert config.get('training.batch_size', 8) == 8
        assert config.get('nonexistent.key', 'default') == 'default'
        print("PASS: test_get_with_default")
    finally:
        os.unlink(config_path)


def test_getattr_access():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'training': {'epochs': 100}}, f)
        config_path = f.name
    
    try:
        config = Config(path=config_path)
        assert config.training.epochs == 100
        print("PASS: test_getattr_access")
    finally:
        os.unlink(config_path)


def test_merge_overrides():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'training': {'epochs': 50, 'batch_size': 8}}, f)
        config_path = f.name
    
    try:
        config = Config(path=config_path)
        config.merge({
            'training': {
                'epochs': 100,
                'learning_rate': 1e-4
            }
        })
        assert config.get('training.epochs') == 100
        assert config.get('training.batch_size') == 8
        assert config.get('training.learning_rate') == 1e-4
        print("PASS: test_merge_overrides")
    finally:
        os.unlink(config_path)


def test_merge_skips_none_values():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'training': {'epochs': 50}}, f)
        config_path = f.name
    
    try:
        config = Config(path=config_path)
        config.merge({
            'training': {
                'epochs': None,
                'batch_size': 8
            }
        })
        assert config.get('training.epochs') == 50
        assert config.get('training.batch_size') == 8
        print("PASS: test_merge_skips_none_values")
    finally:
        os.unlink(config_path)


def test_to_dict():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'training': {'epochs': 50}}, f)
        config_path = f.name
    
    try:
        config = Config(path=config_path)
        d = config.to_dict()
        assert d == {'training': {'epochs': 50}}
        d['training']['epochs'] = 100
        assert config.get('training.epochs') == 50
        print("PASS: test_to_dict")
    finally:
        os.unlink(config_path)


def test_nested_keys():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'model': {
                'encoder': {
                    'name': 'segformer',
                    'hidden_size': 768
                }
            }
        }, f)
        config_path = f.name
    
    try:
        config = Config(path=config_path)
        assert config.get('model.encoder.name') == 'segformer'
        assert config.get('model.encoder.hidden_size') == 768
        print("PASS: test_nested_keys")
    finally:
        os.unlink(config_path)


if __name__ == '__main__':
    test_load_config_from_file()
    test_load_nonexistent_config_raises_error()
    test_invalid_yaml_raises_error()
    test_non_mapping_raises_error()
    test_get_with_default()
    test_getattr_access()
    test_merge_overrides()
    test_merge_skips_none_values()
    test_to_dict()
    test_nested_keys()
    print("\nAll tests passed!")
