#!/usr/bin/env python3
"""
Test script for the Rockland sample1 substudy selection feature.
"""

import pandas as pd
from main import detect_rockland_format, Config

def test_rockland_detection():
    """Test that Rockland format is properly detected."""
    print("Testing Rockland format detection...")
    
    # Test with Rockland format (has all_studies column)
    rockland_columns = ['ursi', 'age', 'sex', 'session_num', 'customID', 'all_studies']
    assert detect_rockland_format(rockland_columns) == True
    print("✅ Rockland format correctly detected")
    
    # Test with non-Rockland format (no all_studies column)
    regular_columns = ['ursi', 'age', 'sex', 'session_num', 'customID']
    assert detect_rockland_format(regular_columns) == False
    print("✅ Non-Rockland format correctly identified")

def test_configuration():
    """Test that configuration constants are properly set."""
    print("\nTesting configuration constants...")
    
    assert hasattr(Config, 'ROCKLAND_BASE_STUDIES')
    assert hasattr(Config, 'DEFAULT_ROCKLAND_STUDIES')
    
    expected_studies = ['Discovery', 'Longitudinal_Adult', 'Longitudinal_Child', 'Neurofeedback']
    assert Config.ROCKLAND_BASE_STUDIES == expected_studies
    assert Config.DEFAULT_ROCKLAND_STUDIES == expected_studies
    print("✅ Configuration constants properly set")

def test_demo_data():
    """Test that our demo data is properly formatted."""
    print("\nTesting demo data...")
    
    demo_path = 'test_rockland_data/demographics.csv'
    df = pd.read_csv(demo_path)
    
    # Check required columns exist
    required_cols = ['ursi', 'age', 'sex', 'session_num', 'customID', 'all_studies']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check that all_studies contains expected substudies
    all_studies_values = df['all_studies'].unique()
    expected_substudies = Config.ROCKLAND_BASE_STUDIES
    
    for studies_str in all_studies_values:
        found_substudies = [study for study in expected_substudies if study in studies_str]
        assert len(found_substudies) > 0, f"No expected substudies found in: {studies_str}"
    
    print("✅ Demo data properly formatted")
    print(f"   Found {len(df)} rows with {len(df['ursi'].unique())} unique participants")
    print(f"   Studies found: {list(all_studies_values)}")

if __name__ == "__main__":
    print("🧪 Testing Rockland Sample1 Substudy Selection Feature\n")
    
    try:
        test_rockland_detection()
        test_configuration()
        test_demo_data()
        print("\n🎉 All tests passed! The Rockland substudy selection feature is ready to use.")
        print("\nTo test the UI:")
        print("streamlit run main.py -- --data-dir test_rockland_data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise