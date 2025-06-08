#!/usr/bin/env python3
"""
Simple test script to verify the rockland-sample1 substudy selection feature.
"""

import os
import sys
import tempfile
import pandas as pd

# Add the current directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import Config, detect_rockland_format

def test_detect_rockland_format():
    """Test the detect_rockland_format function."""
    print("Testing detect_rockland_format()...")
    
    # Test with Rockland columns present
    demographics_columns_with_rockland = ['ursi', 'age', 'sex', 'rockland-sample1']
    result = detect_rockland_format(demographics_columns_with_rockland)
    assert result == True, f"Expected True, got {result}"
    print("✅ Detected Rockland format correctly when column present")
    
    # Test without Rockland columns
    demographics_columns_without_rockland = ['ursi', 'age', 'sex', 'is_DS', 'is_ALG']
    result = detect_rockland_format(demographics_columns_without_rockland)
    assert result == False, f"Expected False, got {result}"
    print("✅ Correctly identified non-Rockland format")
    
    print("detect_rockland_format() tests passed!")

def test_config_constants():
    """Test that the new configuration constants are properly defined."""
    print("Testing configuration constants...")
    
    # Test Rockland constants exist
    assert hasattr(Config, 'ROCKLAND_SAMPLE1_COLUMNS'), "ROCKLAND_SAMPLE1_COLUMNS not defined"
    assert hasattr(Config, 'ROCKLAND_SAMPLE1_LABELS'), "ROCKLAND_SAMPLE1_LABELS not defined"
    assert hasattr(Config, 'DEFAULT_ROCKLAND_SAMPLE1_SELECTION'), "DEFAULT_ROCKLAND_SAMPLE1_SELECTION not defined"
    print("✅ All configuration constants defined")
    
    # Test values are correct
    assert Config.ROCKLAND_SAMPLE1_COLUMNS == ['rockland-sample1'], f"Unexpected columns: {Config.ROCKLAND_SAMPLE1_COLUMNS}"
    assert Config.ROCKLAND_SAMPLE1_LABELS == {'rockland-sample1': 'Rockland Sample 1'}, f"Unexpected labels: {Config.ROCKLAND_SAMPLE1_LABELS}"
    assert Config.DEFAULT_ROCKLAND_SAMPLE1_SELECTION == ['rockland-sample1'], f"Unexpected default: {Config.DEFAULT_ROCKLAND_SAMPLE1_SELECTION}"
    print("✅ Configuration values are correct")
    
    print("Configuration constants tests passed!")

def test_demo_data_creation():
    """Create demo data files to test the feature manually."""
    print("Creating demo data files for manual testing...")
    
    # Create a temporary data directory for testing
    test_data_dir = "test_rockland_data"
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create demographics.csv with rockland-sample1 column
    demographics_data = {
        'ursi': ['URSIxxxxxx01', 'URSIxxxxxx02', 'URSIxxxxxx03', 'URSIxxxxxx04', 'URSIxxxxxx05'],
        'age': [25, 30, 35, 28, 32],
        'sex': [1.0, 2.0, 1.0, 2.0, 1.0],  # 1=Female, 2=Male
        'rockland-sample1': [1, 1, 0, 1, 0]  # 1=included in substudy, 0=not included
    }
    
    demographics_df = pd.DataFrame(demographics_data)
    demographics_df.to_csv(os.path.join(test_data_dir, 'demographics.csv'), index=False)
    print(f"✅ Created demographics.csv with {len(demographics_df)} participants")
    print(f"   - {demographics_df['rockland-sample1'].sum()} participants in rockland-sample1 substudy")
    
    # Create a simple behavioral data file
    behavioral_data = {
        'ursi': ['URSIxxxxxx01', 'URSIxxxxxx02', 'URSIxxxxxx03', 'URSIxxxxxx04', 'URSIxxxxxx05'],
        'task_accuracy': [0.85, 0.92, 0.78, 0.88, 0.90],
        'reaction_time': [450, 420, 480, 440, 430]
    }
    
    behavioral_df = pd.DataFrame(behavioral_data)
    behavioral_df.to_csv(os.path.join(test_data_dir, 'task_performance.csv'), index=False)
    print(f"✅ Created task_performance.csv with behavioral data")
    
    print(f"Demo data created in '{test_data_dir}' directory")
    print("To test manually, run:")
    print(f"streamlit run main.py -- --data-dir {test_data_dir}")
    print("You should see 'Substudy Selection' section with 'Rockland Sample 1' checkbox")
    
    return test_data_dir

def main():
    """Run all tests."""
    print("🧪 Testing Rockland Sample1 Substudy Selection Feature")
    print("=" * 60)
    
    try:
        test_detect_rockland_format()
        print()
        test_config_constants()
        print()
        test_data_dir = test_demo_data_creation()
        print()
        print("🎉 All tests passed!")
        print(f"Test data directory: {test_data_dir}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()