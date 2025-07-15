#!/usr/bin/env python3
"""
Simple test to verify the config fix works
"""

import torch
import torch.nn as nn
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # No config attribute - this should trigger the fix
        
    def forward(self, **kwargs):
        return None

def test_config_fix():
    """Test that the config fix works"""
    print("=== Testing Config Fix ===")
    
    # Create mock model without config
    base_model = MockModel()
    
    try:
        # This should not fail now
        coconut_model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=32001,
            start_latent_id=32002,
            end_latent_id=32003,
            eos_token_id=2
        )
        print("✓ MultimodalCoconut created successfully with mock model")
        print(f"✓ Config type: {type(coconut_model.config)}")
        print(f"✓ Config use_return_dict: {coconut_model.config.use_return_dict}")
        return True
    except Exception as e:
        print(f"✗ Error creating MultimodalCoconut: {e}")
        return False

if __name__ == "__main__":
    success = test_config_fix()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")