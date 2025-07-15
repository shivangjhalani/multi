#!/usr/bin/env python3
"""
Validate the project structure for Multimodal CoCoNuT
"""

from pathlib import Path
import sys

def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists and report the result."""
    file_path = Path(path)
    exists = file_path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def check_directory_structure():
    """Check that all required directories and files exist."""
    print("Checking project structure...")
    print("=" * 50)
    
    required_files = [
        # Core files
        ("requirements.txt", "Requirements file"),
        ("setup.py", "Setup script"),
        ("README.md", "README file"),
        ("run.py", "Main training script"),
        ("test_infrastructure.py", "Infrastructure test"),
        
        # Configuration files
        ("args/multimodal_coconut.yaml", "CoCoNuT config"),
        ("args/multimodal_cot.yaml", "CoT config"),
        
        # Package structure
        ("multimodal_coconut/__init__.py", "Main package init"),
        ("multimodal_coconut/config/__init__.py", "Config package init"),
        ("multimodal_coconut/config/config.py", "Config implementation"),
        ("multimodal_coconut/model/__init__.py", "Model package init"),
        ("multimodal_coconut/model/multimodal_coconut.py", "Model implementation"),
        ("multimodal_coconut/data/__init__.py", "Data package init"),
        ("multimodal_coconut/training/__init__.py", "Training package init"),
        ("multimodal_coconut/utils/__init__.py", "Utils package init"),
        ("multimodal_coconut/utils/distributed.py", "Distributed utils"),
        ("multimodal_coconut/utils/logging.py", "Logging utils"),
        ("multimodal_coconut/utils/misc.py", "Misc utils"),
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    print("=" * 50)
    
    if all_exist:
        print("✓ All required files exist!")
        return True
    else:
        print("✗ Some required files are missing!")
        return False

def check_config_validity():
    """Check that configuration files are valid YAML."""
    print("\nChecking configuration files...")
    print("=" * 50)
    
    try:
        import yaml
        
        config_files = [
            "args/multimodal_coconut.yaml",
            "args/multimodal_cot.yaml"
        ]
        
        all_valid = True
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"✓ Valid YAML: {config_file}")
            except Exception as e:
                print(f"✗ Invalid YAML: {config_file} - {e}")
                all_valid = False
        
        return all_valid
        
    except ImportError:
        print("⚠ PyYAML not installed, skipping YAML validation")
        return True

def main():
    """Run all validation checks."""
    print("Multimodal CoCoNuT Project Structure Validation")
    print("=" * 60)
    
    structure_ok = check_directory_structure()
    config_ok = check_config_validity()
    
    print("\n" + "=" * 60)
    
    if structure_ok and config_ok:
        print("✓ Project structure validation passed!")
        print("The infrastructure is properly set up.")
        return 0
    else:
        print("✗ Project structure validation failed!")
        print("Please check the missing files or errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())