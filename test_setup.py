"""
Quick test script to verify RLHF pipeline setup
"""
import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        return False
    
    try:
        import trl
        print(f"✓ TRL {trl.__version__}")
    except ImportError:
        print("✗ TRL not installed")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError:
        print("✗ Datasets not installed")
        return False
    
    return True

def test_data_files():
    """Test if data files exist"""
    print("\nTesting data files...")
    
    files = [
        "dataset/sample_preference.json",
        "dataset/sample_prompt.json"
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} not found")
            all_exist = False
    
    return all_exist

def test_module_imports():
    """Test if our modules can be imported"""
    print("\nTesting module imports...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from data_loader import PreferenceDataset, PromptDataset
        print("✓ data_loader")
    except Exception as e:
        print(f"✗ data_loader: {e}")
        return False
    
    try:
        from reward_model import RewardModel
        print("✓ reward_model")
    except Exception as e:
        print(f"✗ reward_model: {e}")
        return False
    
    try:
        from rlhf_trainer import RLHFTrainer
        print("✓ rlhf_trainer")
    except Exception as e:
        print(f"✗ rlhf_trainer: {e}")
        return False
    
    try:
        from inference import RLHFInference
        print("✓ inference")
    except Exception as e:
        print(f"✗ inference: {e}")
        return False
    
    try:
        from pipeline import RLHFPipeline
        print("✓ pipeline")
    except Exception as e:
        print(f"✗ pipeline: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("="*50)
    print("RLHF Pipeline Setup Test")
    print("="*50)
    
    results = []
    results.append(("Package Imports", test_imports()))
    results.append(("Data Files", test_data_files()))
    results.append(("Module Imports", test_module_imports()))
    
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! Pipeline is ready to use.")
    else:
        print("\n✗ Some tests failed. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

