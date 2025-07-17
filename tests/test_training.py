import pytest
from multimodal_coconut.config.config import Config
from multimodal_coconut.training.stage_manager import StageManager, StageInfo

@pytest.fixture
def base_config():
    """Fixture for a base configuration for the StageManager."""
    return Config({
        "epochs_per_stage": 2,
        "max_latent_stage": 3,
        "c_thought": 2,
        "uniform_prob": 0.0,  # Disable uniform mixing for predictable tests
        "cot": False,
        "coconut": True,
        "no_cot": False,
        "pad_latent_to_max": False
    })

def test_stage_manager_init(base_config):
    """Test the initialization of the StageManager."""
    manager = StageManager(base_config)
    assert manager.epochs_per_stage == 2
    assert manager.max_latent_stage == 3
    assert manager.c_thought == 2

def test_get_current_stage(base_config):
    """Test the calculation of the current stage based on the epoch."""
    manager = StageManager(base_config)
    assert manager.get_current_stage(epoch=0) == 0
    assert manager.get_current_stage(epoch=1) == 0
    assert manager.get_current_stage(epoch=2) == 1
    assert manager.get_current_stage(epoch=3) == 1
    assert manager.get_current_stage(epoch=4) == 2

def test_get_stage_info(base_config):
    """Test that correct information is returned for each stage."""
    manager = StageManager(base_config)
    
    # Test Stage 0 (CoT)
    info_0 = manager.get_stage_info(0)
    assert isinstance(info_0, StageInfo)
    assert info_0.is_cot_stage is True
    assert info_0.num_latent_tokens == 0

    # Test a CoCoNuT stage
    info_2 = manager.get_stage_info(2)
    assert info_2.is_cot_stage is False
    assert info_2.num_latent_steps == 2
    assert info_2.num_latent_tokens == 4  # 2 (stage) * 2 (c_thought)

    # Test a stage beyond max_latent_stage
    info_4 = manager.get_stage_info(4)
    assert info_4.num_latent_steps == 3  # Capped at max_latent_stage
    assert info_4.num_latent_tokens == 6  # 3 * 2

def test_get_effective_stage_for_sample(base_config):
    """Test the calculation of the effective stage for a sample."""
    manager = StageManager(base_config)
    sample_steps = ["step 1", "step 2", "step 3", "step 4"]

    # Test scheduled_stage = 1
    eff_stage, n_skip, n_latent = manager.get_effective_stage_for_sample(
        scheduled_stage=1, sample_steps=sample_steps
    )
    assert eff_stage == 1
    assert n_skip == 1
    assert n_latent == 2 # 1 * c_thought

    # Test scheduled_stage exceeding max_latent_stage
    eff_stage, n_skip, n_latent = manager.get_effective_stage_for_sample(
        scheduled_stage=4, sample_steps=sample_steps
    )
    assert eff_stage == 4
    assert n_skip == 10000 # Skips all steps
    assert n_latent == 6 # max_latent_stage * c_thought

def test_uniform_mixing(base_config):
    """Test the uniform mixing logic."""
    base_config.uniform_prob = 1.0  # Force uniform mixing
    manager = StageManager(base_config)
    sample_steps = ["step 1", "step 2"]
    
    # Since it's random, we can't assert a specific stage,
    # but we can check if the returned stage is valid.
    possible_stages = {0, 1, 2}
    eff_stage, _, _ = manager.get_effective_stage_for_sample(
        scheduled_stage=2, sample_steps=sample_steps
    )
    assert eff_stage in possible_stages