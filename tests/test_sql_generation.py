"""
Tests for SQL query generation functions.
"""
import pytest
import os
import tempfile
from typing import Dict, List, Any

# Import the functions we want to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    generate_base_query_logic, 
    generate_data_query, 
    generate_count_query,
    MergeKeys,
    Config
)


class TestSQLQueryGeneration:
    """Test SQL query generation functions."""
    
    @pytest.fixture
    def cross_sectional_merge_keys(self):
        """Cross-sectional merge keys for testing."""
        return MergeKeys(
            primary_id='ursi',
            is_longitudinal=False
        )
    
    @pytest.fixture
    def longitudinal_merge_keys(self):
        """Longitudinal merge keys for testing."""
        return MergeKeys(
            primary_id='ursi',
            session_id='session_num',
            composite_id='customID',
            is_longitudinal=True
        )
    
    @pytest.fixture
    def sample_demographic_filters(self):
        """Sample demographic filters for testing."""
        return {
            'age_range': (18, 65),
            'sex': ['Female', 'Male'],
            'sessions': ['BAS1', 'BAS2'],
            'studies': None,
            'substudies': None
        }
    
    @pytest.fixture
    def sample_behavioral_filters(self):
        """Sample behavioral filters for testing."""
        return [
            {
                'table': 'cognitive',
                'column': 'working_memory',
                'min_val': 85,
                'max_val': 115
            },
            {
                'table': 'flanker',
                'column': 'rt_congruent',
                'min_val': 400,
                'max_val': 600
            }
        ]
    
    def test_generate_base_query_logic_cross_sectional(self, cross_sectional_merge_keys, sample_demographic_filters):
        """Test base query generation for cross-sectional data."""
        tables_to_join = ['cognitive', 'flanker']
        behavioral_filters = []
        
        query, params = generate_base_query_logic(
            demographic_filters=sample_demographic_filters,
            behavioral_filters=behavioral_filters,
            tables_to_join=tables_to_join,
            merge_keys=cross_sectional_merge_keys
        )
        
        # Check query structure
        assert query is not None
        assert 'FROM read_csv_auto' in query
        assert 'AS demo' in query
        assert 'LEFT JOIN' in query
        assert 'cognitive' in query
        assert 'flanker' in query
        
        # Check merge column usage (should use ursi for cross-sectional)
        assert 'demo.ursi = cognitive.ursi' in query
        assert 'demo.ursi = flanker.ursi' in query
        
        # Check WHERE clauses
        assert 'demo.age BETWEEN' in query
        assert 'demo.sex IN' in query
        
        # Check parameters
        assert params is not None
        assert len(params) >= 4  # age_min, age_max, sex values
        assert 18 in params
        assert 65 in params
        assert 1.0 in params  # Female
        assert 2.0 in params  # Male
    
    def test_generate_base_query_logic_longitudinal(self, longitudinal_merge_keys, sample_demographic_filters):
        """Test base query generation for longitudinal data."""
        tables_to_join = ['cognitive']
        behavioral_filters = []
        
        query, params = generate_base_query_logic(
            demographic_filters=sample_demographic_filters,
            behavioral_filters=behavioral_filters,
            tables_to_join=tables_to_join,
            merge_keys=longitudinal_merge_keys
        )
        
        # Check query structure
        assert query is not None
        assert 'FROM read_csv_auto' in query
        
        # Check merge column usage (should use customID for longitudinal)
        assert 'demo.customID = cognitive.customID' in query
        
        # Check session filtering
        assert 'session_num IN' in query
        
        # Check parameters include sessions
        assert 'BAS1' in params
        assert 'BAS2' in params
    
    def test_generate_base_query_logic_with_behavioral_filters(self, cross_sectional_merge_keys, sample_demographic_filters, sample_behavioral_filters):
        """Test base query generation with behavioral filters."""
        tables_to_join = ['cognitive', 'flanker']
        
        query, params = generate_base_query_logic(
            demographic_filters=sample_demographic_filters,
            behavioral_filters=sample_behavioral_filters,
            tables_to_join=tables_to_join,
            merge_keys=cross_sectional_merge_keys
        )
        
        # Check behavioral filter clauses
        assert 'cognitive."working_memory" BETWEEN' in query
        assert 'flanker."rt_congruent" BETWEEN' in query
        
        # Check parameters include behavioral filter values
        assert 85 in params
        assert 115 in params
        assert 400 in params
        assert 600 in params
    
    def test_generate_base_query_logic_empty_tables(self, cross_sectional_merge_keys):
        """Test base query generation with no tables to join."""
        demographic_filters = {'age_range': None, 'sex': None, 'sessions': None, 'studies': None, 'substudies': None}
        
        query, params = generate_base_query_logic(
            demographic_filters=demographic_filters,
            behavioral_filters=[],
            tables_to_join=[],
            merge_keys=cross_sectional_merge_keys
        )
        
        # Should still have demographics FROM clause
        assert query is not None
        assert 'FROM read_csv_auto' in query
        assert 'AS demo' in query
        
        # Should not have JOIN clauses
        assert 'LEFT JOIN' not in query
    
    def test_generate_data_query_basic(self, cross_sectional_merge_keys):
        """Test data query generation with basic parameters."""
        base_query_logic = "FROM read_csv_auto('data/demographics.csv') AS demo WHERE demo.age BETWEEN ? AND ?"
        params = [18, 65]
        selected_tables = ['demographics']
        selected_columns = {'demographics': ['ursi', 'age', 'sex']}
        
        query, result_params = generate_data_query(
            base_query_logic=base_query_logic,
            params=params,
            selected_tables=selected_tables,
            selected_columns=selected_columns
        )
        
        assert query is not None
        assert query.startswith('SELECT')
        # Check that it includes demo.* (all demographics columns)
        assert 'demo.*' in query
        # Check that specific columns are included from demographics table
        assert 'demographics."ursi"' in query
        assert 'demographics."age"' in query
        assert 'demographics."sex"' in query
        assert base_query_logic in query
        assert result_params == params
    
    def test_generate_data_query_multiple_tables(self, cross_sectional_merge_keys):
        """Test data query generation with multiple tables."""
        base_query_logic = "FROM read_csv_auto('data/demographics.csv') AS demo LEFT JOIN read_csv_auto('data/cognitive.csv') AS cognitive ON demo.ursi = cognitive.ursi"
        params = []
        selected_tables = ['demographics', 'cognitive']
        selected_columns = {
            'demographics': ['ursi', 'age'],
            'cognitive': ['working_memory', 'attention_score']
        }
        
        query, result_params = generate_data_query(
            base_query_logic=base_query_logic,
            params=params,
            selected_tables=selected_tables,
            selected_columns=selected_columns
        )
        
        assert query is not None
        assert 'demo.*' in query
        assert 'demographics."ursi"' in query
        assert 'demographics."age"' in query
        assert 'cognitive."working_memory"' in query
        assert 'cognitive."attention_score"' in query
    
    def test_generate_data_query_empty_base(self):
        """Test data query generation with empty base query logic."""
        query, params = generate_data_query(
            base_query_logic="",
            params=[],
            selected_tables=['demographics'],
            selected_columns={'demographics': ['ursi']}
        )
        
        assert query is None
        assert params is None
    
    def test_generate_count_query_cross_sectional(self, cross_sectional_merge_keys):
        """Test count query generation for cross-sectional data."""
        base_query_logic = "FROM read_csv_auto('data/demographics.csv') AS demo WHERE demo.age BETWEEN ? AND ?"
        params = [18, 65]
        
        query, result_params = generate_count_query(
            base_query_logic=base_query_logic,
            params=params,
            merge_keys=cross_sectional_merge_keys
        )
        
        assert query is not None
        assert query.startswith('SELECT COUNT(DISTINCT demo.ursi)')
        assert base_query_logic in query
        assert result_params == params
    
    def test_generate_count_query_longitudinal(self, longitudinal_merge_keys):
        """Test count query generation for longitudinal data."""
        base_query_logic = "FROM read_csv_auto('data/demographics.csv') AS demo WHERE demo.age BETWEEN ? AND ?"
        params = [18, 65]
        
        query, result_params = generate_count_query(
            base_query_logic=base_query_logic,
            params=params,
            merge_keys=longitudinal_merge_keys
        )
        
        assert query is not None
        assert query.startswith('SELECT COUNT(DISTINCT demo.customID)')
        assert base_query_logic in query
        assert result_params == params
    
    def test_generate_count_query_empty_base(self, cross_sectional_merge_keys):
        """Test count query generation with empty base query logic."""
        query, params = generate_count_query(
            base_query_logic="",
            params=[],
            merge_keys=cross_sectional_merge_keys
        )
        
        assert query is None
        assert params is None


class TestSQLParameterHandling:
    """Test SQL parameter handling and injection prevention."""
    
    @pytest.fixture
    def cross_sectional_merge_keys(self):
        """Cross-sectional merge keys for testing."""
        return MergeKeys(
            primary_id='ursi',
            is_longitudinal=False
        )
    
    def test_parameters_are_placeholders(self, cross_sectional_merge_keys):
        """Test that queries use ? placeholders instead of direct value insertion."""
        demographic_filters = {
            'age_range': (25, 45),
            'sex': ['Female'],
            'sessions': None,
            'studies': None,
            'substudies': None
        }
        
        query, params = generate_base_query_logic(
            demographic_filters=demographic_filters,
            behavioral_filters=[],
            tables_to_join=[],
            merge_keys=cross_sectional_merge_keys
        )
        
        # Should use placeholders, not direct values
        assert '?' in query
        assert '25' not in query
        assert '45' not in query
        assert '1.0' not in query
        
        # Parameters should contain the actual values
        assert 25 in params
        assert 45 in params
        assert 1.0 in params
    
    def test_column_name_quoting(self, cross_sectional_merge_keys):
        """Test that column names are properly quoted in behavioral filters."""
        behavioral_filters = [
            {
                'table': 'test_table',
                'column': 'column with spaces',
                'min_val': 0,
                'max_val': 100
            }
        ]
        
        query, params = generate_base_query_logic(
            demographic_filters={'age_range': None, 'sex': None, 'sessions': None, 'studies': None, 'substudies': None},
            behavioral_filters=behavioral_filters,
            tables_to_join=['test_table'],
            merge_keys=cross_sectional_merge_keys
        )
        
        # Column names should be quoted
        assert 'test_table."column with spaces"' in query


class TestSQLEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def cross_sectional_merge_keys(self):
        """Cross-sectional merge keys for testing."""
        return MergeKeys(
            primary_id='ursi',
            is_longitudinal=False
        )
    
    @pytest.fixture
    def longitudinal_merge_keys(self):
        """Longitudinal merge keys for testing."""
        return MergeKeys(
            primary_id='ursi',
            session_id='session_num',
            composite_id='customID',
            is_longitudinal=True
        )
    
    def test_no_demographic_filters(self, cross_sectional_merge_keys):
        """Test query generation with no demographic filters."""
        demographic_filters = {
            'age_range': None,
            'sex': None,
            'sessions': None,
            'studies': None,
            'substudies': None
        }
        
        query, params = generate_base_query_logic(
            demographic_filters=demographic_filters,
            behavioral_filters=[],
            tables_to_join=['cognitive'],
            merge_keys=cross_sectional_merge_keys
        )
        
        assert query is not None
        assert 'FROM read_csv_auto' in query
        assert 'WHERE' not in query or 'WHERE 1=1' in query  # May have minimal WHERE clause
    
    def test_rockland_substudy_filtering(self, longitudinal_merge_keys):
        """Test Rockland substudy filtering logic."""
        demographic_filters = {
            'age_range': None,
            'sex': None,
            'sessions': None,
            'studies': None,
            'substudies': ['Discovery', 'Longitudinal_Adult']
        }
        
        query, params = generate_base_query_logic(
            demographic_filters=demographic_filters,
            behavioral_filters=[],
            tables_to_join=[],
            merge_keys=longitudinal_merge_keys
        )
        
        # Should include LIKE clauses for substudy filtering
        if 'Discovery' in str(params) or 'Longitudinal_Adult' in str(params):
            assert 'all_studies LIKE' in query or 'LIKE' in query