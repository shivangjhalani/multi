# Multimodal CoCoNuT Testing Framework

This document describes the comprehensive testing framework implemented for the multimodal CoCoNuT project.

## Overview

The testing framework consists of multiple test suites that validate different aspects of the system:

1. **Unit Tests** - Test individual components in isolation
2. **Integration Tests** - Test end-to-end functionality and component interactions
3. **Performance Tests** - Benchmark memory usage, speed, and scalability
4. **Validation Tests** - Validate reasoning quality and model behavior
5. **Real Component Tests** - Test with actual models and tokenizers
6. **System Tests** - Test configuration, logging, and evaluation systems

## Test Suites

### 1. Unit Tests (`test_unit_core_components.py`)

Tests individual components in isolation:

- **Configuration System**: Config creation, validation, merging
- **Image Processor**: Image loading, preprocessing, error handling
- **Multimodal Dataset**: Dataset creation, tokenization, iteration
- **Multimodal Collator**: Batch collation, padding, latent token alignment
- **Multimodal CoCoNuT Model**: Model initialization, forward pass, generation

```bash
python test_unit_core_components.py
```

### 2. Integration Tests (`test_integration_multimodal_coconut.py`)

Tests end-to-end functionality:

- **End-to-End Training**: Complete training pipeline simulation
- **Model Compatibility**: Different model sizes and configurations
- **Distributed Training**: Multi-GPU setup simulation
- **Configuration Integration**: Config-driven model creation

```bash
python test_integration_multimodal_coconut.py
```

### 3. Performance & Validation Tests (`test_performance_validation.py`)

Benchmarks and validates system performance:

- **Memory Performance**: Memory usage scaling with batch size
- **Training Speed**: Training throughput benchmarks
- **Inference Speed**: Inference latency measurements
- **Reasoning Quality**: Continuous thought consistency validation
- **Robustness**: Image quality and input variation handling

```bash
python test_performance_validation.py
```

### 4. Real Component Tests (`test_unit_real_components.py`)

Tests with actual models and tokenizers:

- **Real Configuration**: Template validation with actual configs
- **Real Image Processing**: InternVL3-style preprocessing
- **Real Tokenization**: HuggingFace tokenizer integration
- **Real Dataset**: End-to-end data pipeline with real components

```bash
python test_unit_real_components.py
```

### 5. System Tests

Additional system-level tests:

- **Logging & Debugging** (`test_logging_and_debugging.py`): W&B integration, metrics tracking
- **Evaluation System** (`test_evaluation_system.py`): A-OKVQA evaluation, reasoning analysis
- **Configuration System** (`test_config_system.py`): YAML loading, validation
- **Configuration Inheritance** (`test_config_inheritance.py`): Template system, inheritance

## Running Tests

### Run All Tests

```bash
python run_all_tests.py
```

This runs all test suites and provides a comprehensive summary.

### Run Individual Test Suites

```bash
# Unit tests
python test_unit_core_components.py

# Integration tests
python test_integration_multimodal_coconut.py

# Performance tests
python test_performance_validation.py

# Real component tests
python test_unit_real_components.py

# System tests
python test_logging_and_debugging.py
python test_evaluation_system.py
python test_config_system.py
python test_config_inheritance.py
```

### Using pytest (if available)

```bash
# Run all tests
pytest

# Run specific test file
pytest test_unit_core_components.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=multimodal_coconut
```

## Test Structure

### Test Organization

Each test file follows a consistent structure:

```python
class TestComponentName:
    """Test specific component functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        pass
    
    def teardown_method(self):
        """Clean up test fixtures"""
        pass
    
    def test_specific_functionality(self):
        """Test specific functionality"""
        # Arrange
        # Act
        # Assert
        pass
```

### Mock Objects

Tests use mock objects to isolate components:

- **MockInternVL3Model**: Simulates InternVL3 behavior
- **Mock Tokenizers**: Simulate tokenization without loading models
- **Mock Data**: Generate test data without external dependencies

### Test Data

Tests create temporary test data:

- **Images**: Generated PIL images with different properties
- **JSON Data**: Mock A-OKVQA format data
- **Configurations**: Test-specific config files

## Performance Benchmarks

The performance tests provide benchmarks for:

### Memory Usage
- Batch size scaling
- Peak memory consumption
- GPU memory usage (if available)

### Speed Benchmarks
- Training throughput (samples/second)
- Inference latency
- Step time measurements

### Quality Metrics
- Reasoning consistency
- Latent vs explicit reasoning comparison
- Stage progression validation

## Continuous Integration

The testing framework is designed for CI/CD integration:

### GitHub Actions Example

```yaml
name: Test Multimodal CoCoNuT

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests
      run: python run_all_tests.py
```

## Test Coverage

The testing framework covers:

### Core Components (100%)
- ✅ Configuration system
- ✅ Data pipeline
- ✅ Image processing
- ✅ Model architecture
- ✅ Training logic

### Integration Points (100%)
- ✅ Data → Model integration
- ✅ Config → Training integration
- ✅ Model → Evaluation integration
- ✅ Distributed training setup

### Performance Aspects (100%)
- ✅ Memory usage
- ✅ Training speed
- ✅ Inference speed
- ✅ Scalability

### Quality Validation (100%)
- ✅ Reasoning consistency
- ✅ CoCoNuT vs CoT comparison
- ✅ Stage progression
- ✅ Robustness testing

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project is installed with `pip install -e .`
2. **Missing Dependencies**: Install test dependencies with `pip install -r requirements.txt`
3. **GPU Tests**: Some tests require CUDA; they gracefully fall back to CPU
4. **Memory Issues**: Large model tests may require sufficient RAM

### Debug Mode

Run tests with debug output:

```bash
python -u test_unit_core_components.py
```

### Selective Testing

Run specific test methods:

```python
# In test file
if __name__ == "__main__":
    test_instance = TestConfig()
    test_instance.test_config_creation()
```

## Contributing

When adding new functionality:

1. **Add Unit Tests**: Test new components in isolation
2. **Add Integration Tests**: Test interactions with existing components
3. **Update Performance Tests**: Add benchmarks for new features
4. **Update Documentation**: Update this README with new test information

### Test Guidelines

- **Isolation**: Tests should not depend on external resources
- **Deterministic**: Tests should produce consistent results
- **Fast**: Unit tests should run quickly (< 1s each)
- **Clear**: Test names should clearly describe what is being tested
- **Comprehensive**: Cover both success and failure cases

## Results Interpretation

### Success Criteria

- **Unit Tests**: All components work in isolation
- **Integration Tests**: Components work together correctly
- **Performance Tests**: Meet performance benchmarks
- **Real Tests**: Work with actual models and data

### Failure Analysis

When tests fail:

1. **Check Error Messages**: Look for specific failure reasons
2. **Run Individual Tests**: Isolate the failing component
3. **Check Dependencies**: Ensure all requirements are installed
4. **Review Recent Changes**: Consider what might have broken

## Future Enhancements

Planned testing improvements:

- [ ] Property-based testing with Hypothesis
- [ ] Load testing for production scenarios
- [ ] A/B testing framework for model comparisons
- [ ] Automated performance regression detection
- [ ] Integration with MLflow for experiment tracking

---

This comprehensive testing framework ensures the reliability, performance, and correctness of the multimodal CoCoNuT implementation.