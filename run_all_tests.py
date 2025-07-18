#!/usr/bin/env python3
"""
Comprehensive test runner for multimodal CoCoNuT

This script runs all test suites:
- Unit tests for core components
- Integration tests for end-to-end functionality
- Performance and validation tests
- Real component tests with actual models
"""

import sys
import subprocess
import time
from pathlib import Path

def run_test_suite(test_file: str, description: str) -> bool:
    """Run a test suite and return success status"""
    print(f"\n{'='*60}")
    print(f"RUNNING {description.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, 
                              text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {description} PASSED (took {duration:.1f}s)")
            return True
        else:
            print(f"\n❌ {description} FAILED (took {duration:.1f}s)")
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n💥 {description} CRASHED: {e} (took {duration:.1f}s)")
        return False

def main():
    """Run all test suites"""
    print("🧪 MULTIMODAL COCONUT COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("This will run all test suites to validate the implementation")
    print("=" * 60)
    
    # Define test suites
    test_suites = [
        ("test_unit_core_components.py", "Unit Tests for Core Components"),
        ("test_integration_multimodal_coconut.py", "Integration Tests"),
        ("test_performance_validation.py", "Performance and Validation Tests"),
        ("test_unit_real_components.py", "Real Component Tests"),
        ("test_logging_and_debugging.py", "Logging and Debugging Tests"),
        ("test_evaluation_system.py", "Evaluation System Tests"),
        ("test_config_system.py", "Configuration System Tests"),
        ("test_config_inheritance.py", "Configuration Inheritance Tests"),
    ]
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Run each test suite
    for test_file, description in test_suites:
        if Path(test_file).exists():
            success = run_test_suite(test_file, description)
            results[description] = success
        else:
            print(f"\n⚠️  {description} - Test file not found: {test_file}")
            results[description] = False
    
    # Summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    print(f"Total test suites: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Success rate: {passed_count/total_count:.1%}")
    print(f"Total time: {total_duration:.1f}s")
    
    print(f"\nDetailed results:")
    for description, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {description}")
    
    # Overall result
    if passed_count == total_count:
        print(f"\n🎉 ALL TESTS PASSED! The multimodal CoCoNuT implementation is ready!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test suite(s) failed. Please review the failures above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())