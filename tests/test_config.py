import os
import yaml
import pytest
from pathlib import Path

from multimodal_coconut.config.config import Config, load_config, validate_config, update_config_for_stage

@pytest.fixture
def temp_config_file(tmp_path):
    config_data = {
        'model_id': 'test_model',
        'c_thought': 2,
        'max_latent_stage': 4,
        'batch_size_training': 8,
        'learning_rate': 1e-5,
        'train_data_path': str(tmp_path / 'train.json')
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    (tmp_path / 'train.json').touch()
    
    return config_path

def test_config_class():
    """Test that the Config class correctly converts a dictionary to attributes."""
    config_dict = {'a': 1, 'b': {'c': 2}}
    config = Config(config_dict)
    assert config.a == 1
    assert config.b['c'] == 2
    assert config.get('a') == 1
    assert config.get('d', 5) == 5

def test_load_config(temp_config_file):
    """Test loading a valid YAML configuration file."""
    config = load_config(str(temp_config_file))
    assert config.model_id == 'test_model'
    assert config.c_thought == 2

def test_load_config_with_env_var(tmp_path):
    """Test loading a config with environment variable substitution."""
    os.environ['TEST_MODEL_ID'] = 'env_model'
    config_content = "model_id: ${TEST_MODEL_ID}\nc_thought: 3"
    config_path = tmp_path / "env_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    config = load_config(str(config_path))
    assert config.model_id == 'env_model'
    assert config.c_thought == 3
    del os.environ['TEST_MODEL_ID']

def test_validate_config_valid(temp_config_file):
    """Test that a valid configuration passes validation."""
    config = load_config(str(temp_config_file))
    try:
        validate_config(config)
    except ValueError:
        pytest.fail("validate_config raised ValueError unexpectedly!")

def test_validate_config_missing_field(temp_config_file):
    """Test that validation fails if a required field is missing."""
    config = load_config(str(temp_config_file))
    delattr(config, 'model_id')
    with pytest.raises(ValueError, match="Missing required config field: model_id"):
        validate_config(config)

def test_validate_config_invalid_value(temp_config_file):
    """Test that validation fails for invalid configuration values."""
    config = load_config(str(temp_config_file))
    config.c_thought = -1
    with pytest.raises(ValueError, match="c_thought must be >= 1"):
        validate_config(config)

def test_validate_config_invalid_type(temp_config_file):
    """Test that validation fails for invalid configuration types."""
    config = load_config(str(temp_config_file))
    config.c_thought = 'abc'
    with pytest.raises(ValueError, match="c_thought must be an integer"):
        validate_config(config)

def test_update_config_for_stage():
    """Test that the config is correctly updated for different training stages."""
    base_config = Config({'coconut': False, 'cot': True})
    
    # Test Stage 0 (CoT)
    stage_0_config = update_config_for_stage(base_config, 0)
    assert stage_0_config.cot is True
    assert stage_0_config.coconut is False
    
    # Test Stage 1 (CoCoNuT)
    stage_1_config = update_config_for_stage(base_config, 1)
    assert stage_1_config.cot is False
    assert stage_1_config.coconut is True