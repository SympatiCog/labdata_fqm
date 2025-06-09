import argparse
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
import streamlit as st


# --- Merge Strategy Classes ---
@dataclass
class MergeKeys:
    """Encapsulates the merge keys for a dataset."""
    primary_id: str  # e.g., 'ursi', 'subject_id'
    session_id: Optional[str] = None  # e.g., 'session_num'
    composite_id: Optional[str] = None  # e.g., 'customID' (derived)
    is_longitudinal: bool = False

    def get_merge_column(self) -> str:
        """Returns the appropriate column for merge operations."""
        if self.is_longitudinal:
            return self.composite_id if self.composite_id else self.primary_id
        return self.primary_id

class MergeStrategy(ABC):
    """Abstract base class for merge strategies."""

    @abstractmethod
    def detect_structure(self, demographics_path: str) -> MergeKeys:
        """Detect the merge structure from demographics file."""
        pass

    @abstractmethod
    def prepare_datasets(self, data_dir: str, merge_keys: MergeKeys) -> bool:
        """Prepare datasets with appropriate merge keys."""
        pass

class FlexibleMergeStrategy(MergeStrategy):
    """Flexible merge strategy that adapts to cross-sectional or longitudinal data."""

    def __init__(self, primary_id_column: str = 'ursi', session_column: str = 'session_num', composite_id_column: str = 'customID'):
        self.primary_id_column = primary_id_column
        self.session_column = session_column
        self.composite_id_column = composite_id_column

    def detect_structure(self, demographics_path: str) -> MergeKeys:
        """Detect whether data is cross-sectional or longitudinal."""
        try:
            # Check if file exists first
            if not os.path.exists(demographics_path):
                raise FileNotFoundError(f"Demographics file not found: {demographics_path}")

            # Read just the headers to check structure
            df_headers = pd.read_csv(demographics_path, nrows=0)
            columns = df_headers.columns.tolist()

            # Check if we have both primary ID and session columns
            has_primary_id = self.primary_id_column in columns
            has_session_id = self.session_column in columns
            has_composite_id = self.composite_id_column in columns

            if has_primary_id and has_session_id:
                # Longitudinal format with separate columns (preferred)
                # Use existing customID if present, otherwise will be created
                return MergeKeys(
                    primary_id=self.primary_id_column,
                    session_id=self.session_column,
                    composite_id=self.composite_id_column,
                    is_longitudinal=True
                )
            elif has_primary_id:
                # Cross-sectional format - use primary ID directly
                return MergeKeys(
                    primary_id=self.primary_id_column,
                    is_longitudinal=False
                )
            elif has_composite_id:
                # Legacy customID-only format (fallback)
                return MergeKeys(
                    primary_id=self.composite_id_column,
                    is_longitudinal=False
                )
            else:
                # Fallback - look for any ID-like column
                id_candidates = [col for col in columns if 'id' in col.lower() or 'ursi' in col.lower()]
                if id_candidates:
                    return MergeKeys(
                        primary_id=id_candidates[0],
                        is_longitudinal=False
                    )
                else:
                    raise ValueError(f"No suitable ID column found in {demographics_path}")

        except (FileNotFoundError, pd.errors.EmptyDataError):
            # Re-raise these specific errors to be handled by caller
            raise
        except Exception as e:
            # Only show error in UI if we're in a Streamlit context
            try:
                st.error(f"Error detecting merge structure: {e}")
            except:
                pass  # Not in Streamlit context
            # Fallback to customID
            return MergeKeys(
                primary_id='customID',
                is_longitudinal=False
            )

    def prepare_datasets(self, data_dir: str, merge_keys: MergeKeys) -> tuple[bool, list[str]]:
        """Prepare datasets with composite IDs if longitudinal."""
        actions_taken = []

        if not merge_keys.is_longitudinal:
            return True, actions_taken  # No preparation needed for cross-sectional

        try:
            # For longitudinal data, ensure all CSV files have composite IDs
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

            for csv_file in csv_files:
                file_path = os.path.join(data_dir, csv_file)
                action = self._add_composite_id_if_needed(file_path, merge_keys)
                if action:
                    actions_taken.append(action)

            return True, actions_taken

        except Exception as e:
            st.error(f"Error preparing longitudinal datasets: {e}")
            return False, actions_taken

    def _add_composite_id_if_needed(self, file_path: str, merge_keys: MergeKeys) -> Optional[str]:
        """Add composite ID column to a file if it doesn't exist or validate existing one."""
        filename = os.path.basename(file_path)

        try:
            df = pd.read_csv(file_path)

            # Skip if we don't have both required columns for longitudinal data
            if (merge_keys.primary_id not in df.columns or
                merge_keys.session_id not in df.columns):
                return None

            # Check if composite ID already exists
            if merge_keys.composite_id in df.columns:
                # Validate existing composite ID consistency
                expected_composite = (
                    df[merge_keys.primary_id].astype(str) + '_' +
                    df[merge_keys.session_id].astype(str)
                )

                # Check if existing composite IDs match expected format
                current_composite = df[merge_keys.composite_id].astype(str)
                if not current_composite.equals(expected_composite):
                    # Inconsistent composite ID - regenerate it
                    df[merge_keys.composite_id] = expected_composite
                    df.to_csv(file_path, index=False)
                    return f"🔧 Fixed inconsistent customID in {filename}"
                # else: composite ID is correct, no action needed
                return None
            else:
                # Create new composite ID
                df[merge_keys.composite_id] = (
                    df[merge_keys.primary_id].astype(str) + '_' +
                    df[merge_keys.session_id].astype(str)
                )
                df.to_csv(file_path, index=False)
                return f"✅ Added customID to {filename}"

        except Exception as e:
            # Log warning but don't fail - this file might not need composite ID
            return f"⚠️ Could not process {filename}: {str(e)}"

# --- Configuration ---
class Config:
    # File and directory settings
    DATA_DIR = 'data'
    DEMOGRAPHICS_FILE = 'demographics.csv'
    PARTICIPANT_ID_COLUMN = 'customID'  # Legacy default, will be auto-detected
    PRIMARY_ID_COLUMN = 'ursi'  # User-specified primary ID (subject identifier)
    SESSION_COLUMN = 'session_num'  # Session identifier for longitudinal data
    COMPOSITE_ID_COLUMN = 'customID'  # Composite ID column name for longitudinal data

    # Merge strategy instance
    _merge_strategy: Optional['FlexibleMergeStrategy'] = None
    _merge_keys: Optional['MergeKeys'] = None

    # UI defaults
    DEFAULT_AGE_RANGE = (0, 120)
    DEFAULT_AGE_SELECTION = (18, 80)
    DEFAULT_FILTER_RANGE = (0, 100)
    MAX_DISPLAY_ROWS = 50
    CACHE_TTL_SECONDS = 600

    # Sex mapping for demographics
    SEX_MAPPING = {
        'Female': 1.0,
        'Male': 2.0,
        'Other': 3.0,
        'Unspecified': 0.0
    }

    # Available sex options for UI
    SEX_OPTIONS = ['Female', 'Male', 'Other', 'Unspecified']
    DEFAULT_SEX_SELECTION = ['Female', 'Male']

    # RS1 study columns for study selection
    RS1_STUDY_COLUMNS = ['is_DS', 'is_ALG', 'is_CLG', 'is_NFB']
    RS1_STUDY_LABELS = {'is_DS': 'DS Study', 'is_ALG': 'ALG Study', 'is_CLG': 'CLG Study', 'is_NFB': 'NFB Study'}
    DEFAULT_STUDY_SELECTION = ['is_DS', 'is_ALG', 'is_CLG', 'is_NFB']

    # Rockland sample1 substudy configuration
    ROCKLAND_BASE_STUDIES = ['Discovery', 'Longitudinal_Adult', 'Longitudinal_Child', 'Neurofeedback']
    DEFAULT_ROCKLAND_STUDIES = ['Discovery', 'Longitudinal_Adult', 'Longitudinal_Child', 'Neurofeedback']

    # Session selection options
    SESSION_OPTIONS = ['BAS1', 'BAS2', 'BAS3', 'FLU1', 'FLU2', 'FLU3', 'NFB', 'TRT', 'TRT2']
    DEFAULT_SESSION_SELECTION = ['BAS1', 'BAS2', 'BAS3', 'FLU1', 'FLU2', 'FLU3', 'NFB', 'TRT', 'TRT2']

    # Rockland sample1 substudy columns
    ROCKLAND_SAMPLE1_COLUMNS = ['rockland-sample1']
    ROCKLAND_SAMPLE1_LABELS = {'rockland-sample1': 'Rockland Sample 1'}
    DEFAULT_ROCKLAND_SAMPLE1_SELECTION = ['rockland-sample1']

    @classmethod
    def get_demographics_table_name(cls) -> str:
        return cls.DEMOGRAPHICS_FILE.replace('.csv', '')

    @classmethod
    def get_merge_strategy(cls) -> 'FlexibleMergeStrategy':
        """Get or create the merge strategy instance."""
        if cls._merge_strategy is None:
            cls._merge_strategy = FlexibleMergeStrategy(
                primary_id_column=cls.PRIMARY_ID_COLUMN,
                session_column=cls.SESSION_COLUMN,
                composite_id_column=cls.COMPOSITE_ID_COLUMN
            )
        return cls._merge_strategy

    @classmethod
    def get_merge_keys(cls) -> 'MergeKeys':
        """Get or detect the merge keys for the current dataset."""
        if cls._merge_keys is None:
            demographics_path = os.path.join(cls.DATA_DIR, cls.DEMOGRAPHICS_FILE)
            try:
                cls._merge_keys = cls.get_merge_strategy().detect_structure(demographics_path)
                # Update PARTICIPANT_ID_COLUMN to match detected structure
                cls.PARTICIPANT_ID_COLUMN = cls._merge_keys.get_merge_column()
            except (FileNotFoundError, pd.errors.EmptyDataError, Exception):
                # Handle empty data directory gracefully - provide reasonable defaults
                cls._merge_keys = MergeKeys(
                    primary_id=cls.PRIMARY_ID_COLUMN,
                    session_id=cls.SESSION_COLUMN,
                    composite_id=cls.COMPOSITE_ID_COLUMN,
                    is_longitudinal=True  # Default to longitudinal for flexibility
                )
                cls.PARTICIPANT_ID_COLUMN = cls._merge_keys.get_merge_column()
        return cls._merge_keys

    @classmethod
    def refresh_merge_detection(cls) -> None:
        """Force re-detection of merge structure."""
        cls._merge_keys = None
        cls._merge_strategy = None

    @classmethod
    def parse_cli_args(cls) -> None:
        """Parse command line arguments and update Config class attributes."""
        parser = argparse.ArgumentParser(
            description='TBS Data Query and Merge Tool - Flexible cross-sectional and longitudinal data browser',
            epilog='''
Examples:
  # Auto-detect column names (default):
  streamlit run main.py

  # Common research data formats:
  streamlit run main.py -- --primary-id-column subject_id --session-column timepoint
  streamlit run main.py -- --primary-id-column participant_id --session-column visit
  streamlit run main.py -- --primary-id-column custom_ID --session-column session_num
  streamlit run main.py -- --primary-id-column SubjectID --session-column Session

  # Custom data directory:
  streamlit run main.py -- --data-dir newdata --primary-id-column custom_ID --session-column session_num
            ''',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        # File and directory settings
        parser.add_argument(
            '--data-dir',
            type=str,
            default=cls.DATA_DIR,
            help='Directory containing CSV data files'
        )
        parser.add_argument(
            '--demographics-file',
            type=str,
            default=cls.DEMOGRAPHICS_FILE,
            help='Demographics CSV filename'
        )
        parser.add_argument(
            '--participant-id-column',
            type=str,
            default=cls.PARTICIPANT_ID_COLUMN,
            help='[LEGACY] Forces specific merge column, disables auto-detection. Use --primary-id-column instead.'
        )
        parser.add_argument(
            '--primary-id-column',
            type=str,
            default=cls.PRIMARY_ID_COLUMN,
            help='[RECOMMENDED] Primary subject ID column for auto-detection. Examples: ursi, subject_id, participant_id, custom_ID'
        )
        parser.add_argument(
            '--session-column',
            type=str,
            default=cls.SESSION_COLUMN,
            help='Session identifier column for longitudinal data. Common names: session_num, session, timepoint, visit'
        )
        parser.add_argument(
            '--composite-id-column',
            type=str,
            default='customID',
            help='Name for composite ID column (created from primary_id + session for longitudinal data)'
        )

        # UI defaults
        parser.add_argument(
            '--max-display-rows',
            type=int,
            default=cls.MAX_DISPLAY_ROWS,
            help='Maximum rows to display in preview'
        )
        parser.add_argument(
            '--cache-ttl-seconds',
            type=int,
            default=cls.CACHE_TTL_SECONDS,
            help='Cache time-to-live in seconds'
        )
        parser.add_argument(
            '--default-age-min',
            type=int,
            default=cls.DEFAULT_AGE_SELECTION[0],
            help='Default minimum age for age filter'
        )
        parser.add_argument(
            '--default-age-max',
            type=int,
            default=cls.DEFAULT_AGE_SELECTION[1],
            help='Default maximum age for age filter'
        )

        # Determine if we should parse arguments
        should_parse = False
        our_args = sys.argv[1:]

        # Check if running via Streamlit with -- separator
        if '--' in sys.argv:
            dash_index = sys.argv.index('--')
            our_args = sys.argv[dash_index + 1:]
            should_parse = True
        # Check if running directly (not via streamlit run)
        elif len(sys.argv) > 0 and 'streamlit' not in sys.argv[0]:
            should_parse = True

        if should_parse and our_args:
            try:
                args = parser.parse_args(our_args)

                # Update class attributes
                cls.DATA_DIR = args.data_dir
                cls.DEMOGRAPHICS_FILE = args.demographics_file
                cls.PARTICIPANT_ID_COLUMN = args.participant_id_column
                cls.PRIMARY_ID_COLUMN = args.primary_id_column
                cls.SESSION_COLUMN = args.session_column
                cls.COMPOSITE_ID_COLUMN = args.composite_id_column
                cls.MAX_DISPLAY_ROWS = args.max_display_rows
                cls.CACHE_TTL_SECONDS = args.cache_ttl_seconds
                cls.DEFAULT_AGE_SELECTION = (args.default_age_min, args.default_age_max)
                # Force re-detection with new settings
                cls.refresh_merge_detection()
            except SystemExit:
                # Handle --help or invalid arguments gracefully
                pass


# --- Page Setup ---
st.set_page_config(
    page_title="The Basic Scientist's Data Query Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- File Upload Helper Functions ---

def secure_filename(filename: str) -> str:
    """
    Sanitize filename for security by removing path components and dangerous characters.
    Replaces whitespace and special characters with underscores to avoid SQL syntax errors.
    """
    # Remove path components and keep only the basename
    filename = os.path.basename(filename)

    # Replace whitespace with underscores first
    filename = re.sub(r'\s+', '_', filename)

    # Remove or replace remaining dangerous characters, keep alphanumeric, dots, hyphens, underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)

    # Remove leading/trailing underscores
    filename = filename.strip('_')

    # Limit length to prevent filesystem issues
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext

    return filename

def validate_csv_file(uploaded_file, required_columns: Optional[list[str]] = None) -> tuple[list[str], Optional[pd.DataFrame]]:
    """
    Validate uploaded CSV file and return any errors along with the DataFrame if valid.

    Args:
        uploaded_file: Streamlit uploaded file object
        required_columns: List of column names that must be present

    Returns:
        Tuple of (list of error messages, DataFrame or None)
    """
    errors = []

    try:
        # File size check (50MB limit)
        if uploaded_file.size > 50 * 1024 * 1024:
            errors.append("File too large (maximum 50MB)")

        # File extension check
        if not uploaded_file.name.lower().endswith('.csv'):
            errors.append("File must be a CSV (.csv extension)")

        # Try to read the CSV
        df = pd.read_csv(uploaded_file)

        # Check for empty files
        if len(df) == 0:
            errors.append("File is empty (no data rows)")

        # Check for required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing required columns: {', '.join(missing_cols)}")

        # Check for reasonable number of columns
        if len(df.columns) == 0:
            errors.append("File has no columns")
        elif len(df.columns) > 1000:
            errors.append("File has too many columns (maximum 1000)")

        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
            errors.append(f"Duplicate column names found: {', '.join(set(duplicates))}")

        return errors, df if not errors else None

    except pd.errors.EmptyDataError:
        errors.append("File is empty or contains no valid CSV data")
    except pd.errors.ParserError as e:
        errors.append(f"Invalid CSV format: {str(e)}")
    except UnicodeDecodeError:
        errors.append("File encoding not supported (please use UTF-8)")
    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")

    return errors, None

def save_uploaded_files_to_data_dir(uploaded_files, data_dir: str) -> list[str]:
    """
    Save uploaded files to the data directory with conflict resolution.

    Args:
        uploaded_files: List of validated uploaded file objects
        data_dir: Target directory for saving files

    Returns:
        List of successfully saved file paths
    """
    saved_files = []

    # Ensure data directory exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        try:
            # Sanitize filename
            safe_filename = secure_filename(uploaded_file.name)
            file_path = os.path.join(data_dir, safe_filename)

            # Handle filename conflicts by adding a number suffix
            if os.path.exists(file_path):
                base_name, ext = os.path.splitext(safe_filename)
                counter = 1
                while os.path.exists(file_path):
                    new_name = f"{base_name}_{counter}{ext}"
                    file_path = os.path.join(data_dir, new_name)
                    counter += 1

                st.warning(f"File '{uploaded_file.name}' already exists, saved as '{os.path.basename(file_path)}'")

            # Save file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            saved_files.append(file_path)
            st.success(f"✅ Saved '{uploaded_file.name}' ({uploaded_file.size:,} bytes)")

        except Exception as e:
            st.error(f"❌ Failed to save '{uploaded_file.name}': {str(e)}")

    return saved_files

# --- Helper Functions ---

@st.cache_resource
def get_db_connection() -> duckdb.DuckDBPyConnection:
    """
    Establishes and caches a connection to an in-memory DuckDB database.
    """
    return duckdb.connect(database=':memory:', read_only=False)

def scan_csv_files(data_dir: str) -> list[str]:
    """
    Scans directory for CSV files and returns list of filenames.
    """
    try:
        files = os.listdir(data_dir)
        return [f for f in files if f.endswith('.csv')]
    except FileNotFoundError:
        st.error(f"Error: The data directory was not found at '{data_dir}'.")
        return []
    except PermissionError:
        st.error(f"Error: Permission denied accessing directory '{data_dir}'.")
        return []
    except OSError as e:
        st.error(f"Error accessing directory '{data_dir}': {e}")
        return []

def get_table_alias(table_name: str) -> str:
    """
    Returns the appropriate alias for a table (demo vs table name).
    """
    return 'demo' if table_name == Config.get_demographics_table_name() else table_name

def is_numeric_column(dtype_str: str) -> bool:
    """
    Checks if a column data type is numeric.
    """
    return 'int' in dtype_str or 'float' in dtype_str

def detect_rs1_format(demographics_columns: list[str]) -> bool:
    """
    Detects if the demographics file is in RS1 format by checking for study columns.
    """
    return all(col in demographics_columns for col in Config.RS1_STUDY_COLUMNS)

def detect_rockland_format(demographics_columns: list[str]) -> bool:
    """
    Detects if the demographics file is in Rockland sample1 format by checking for all_studies column.
    """
    return 'all_studies' in demographics_columns

def get_unique_session_values(data_dir: str, merge_keys: MergeKeys) -> list[str]:
    """
    Extract unique session values from the detected session column across all CSV files.
    """
    if not merge_keys.is_longitudinal or not merge_keys.session_id:
        return []

    unique_sessions = set()

    try:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            try:
                # Read just the session column to get unique values
                df = pd.read_csv(file_path, usecols=[merge_keys.session_id] if merge_keys.session_id in pd.read_csv(file_path, nrows=0).columns else None)
                if merge_keys.session_id in df.columns:
                    # Convert to string and remove NaN values
                    sessions = df[merge_keys.session_id].dropna().astype(str).unique()
                    unique_sessions.update(sessions)
            except Exception:
                # Skip files that don't have the session column or can't be read
                continue

    except Exception:
        # If we can't scan files, fall back to empty list
        return []

    # Sort the sessions for consistent ordering
    session_list = sorted(unique_sessions)
    return session_list

def validate_csv_structure(file_path: str, filename: str, merge_keys: MergeKeys) -> bool:
    """
    Validates basic CSV structure and required columns based on data format.
    """
    try:
        # Check if file can be read
        df_headers = pd.read_csv(file_path, nrows=0)
        columns = df_headers.columns.tolist()

        if len(columns) == 0:
            st.warning(f"Warning: '{filename}' has no columns.")
            return False

        # For cross-sectional data, require primary ID
        if not merge_keys.is_longitudinal:
            if merge_keys.primary_id not in columns:
                st.warning(f"Warning: '{filename}' missing required column '{merge_keys.primary_id}'.")
                return False
        else:
            # For longitudinal data, we can be more flexible:
            # - Prefer composite ID if it exists
            # - Otherwise require primary ID (composite will be created)
            has_composite = merge_keys.composite_id in columns
            has_primary = merge_keys.primary_id in columns

            if not has_composite and not has_primary:
                st.warning(f"Warning: '{filename}' missing required column '{merge_keys.primary_id}' or '{merge_keys.composite_id}'.")
                return False

        return True

    except Exception as e:
        st.error(f"Error validating '{filename}': {e}")
        return False

def extract_column_metadata_fast(file_path: str, table_name: str, is_demo_table: bool, merge_keys: MergeKeys) -> tuple[list[str], dict[str, str]]:
    """
    Extracts column information and data types from CSV without loading full data.
    """
    df_name = get_table_alias(table_name if not is_demo_table else Config.get_demographics_table_name())

    # Read only first few rows for metadata
    df_sample = pd.read_csv(file_path, nrows=100)

    # Exclude ID columns from the available columns list
    id_columns_to_exclude = {merge_keys.primary_id}
    if merge_keys.session_id:
        id_columns_to_exclude.add(merge_keys.session_id)
    if merge_keys.composite_id and merge_keys.composite_id in df_sample.columns:
        id_columns_to_exclude.add(merge_keys.composite_id)

    columns = [col for col in df_sample.columns if col not in id_columns_to_exclude]

    column_dtypes = {}
    for col in df_sample.columns:
        if col in id_columns_to_exclude:
            continue
        col_key = f"{df_name}.{col}"
        column_dtypes[col_key] = str(df_sample[col].dtype)

    return columns, column_dtypes

def calculate_numeric_ranges_fast(file_path: str, table_name: str, is_demo_table: bool, column_dtypes: dict[str, str], merge_keys: MergeKeys) -> dict[str, tuple[float, float]]:
    """
    Calculates min/max ranges for numeric columns using chunked reading.
    """
    df_name = get_table_alias(table_name if not is_demo_table else Config.get_demographics_table_name())
    column_ranges = {}

    # ID columns to exclude from numeric analysis
    id_columns_to_exclude = {merge_keys.primary_id}
    if merge_keys.session_id:
        id_columns_to_exclude.add(merge_keys.session_id)
    if merge_keys.composite_id:
        id_columns_to_exclude.add(merge_keys.composite_id)

    # Get numeric columns from dtypes
    numeric_cols = []
    for col_key, dtype_str in column_dtypes.items():
        if col_key.startswith(f"{df_name}.") and is_numeric_column(dtype_str):
            col_name = col_key.split('.', 1)[1]
            if col_name not in id_columns_to_exclude:
                numeric_cols.append(col_name)

    if not numeric_cols:
        return column_ranges

    # Read file in chunks to calculate ranges efficiently
    try:
        chunk_iter = pd.read_csv(file_path, chunksize=1000, usecols=numeric_cols)

        min_vals = {col: float('inf') for col in numeric_cols}
        max_vals = {col: float('-inf') for col in numeric_cols}

        for chunk in chunk_iter:
            for col in numeric_cols:
                if col in chunk.columns:
                    # Coerce to numeric, invalid parsing will be set as NaN
                    numeric_series = pd.to_numeric(chunk[col], errors='coerce')

                    # Get min and max, ignoring NaN values
                    col_min = numeric_series.min()
                    col_max = numeric_series.max()

                    if pd.notna(col_min):
                        min_vals[col] = min(min_vals[col], col_min)
                    if pd.notna(col_max):
                        max_vals[col] = max(max_vals[col], col_max)

        # Store results
        for col in numeric_cols:
            if min_vals[col] != float('inf') and max_vals[col] != float('-inf'):
                col_key = f"{df_name}.{col}"
                column_ranges[col_key] = (float(min_vals[col]), float(max_vals[col]))

    except Exception:
        # Fallback to full read if chunked reading fails
        df = pd.read_csv(file_path)
        for col in numeric_cols:
            if col in df.columns:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                min_val = numeric_series.min()
                max_val = numeric_series.max()
                if pd.notna(min_val) and pd.notna(max_val):
                    col_key = f"{df_name}.{col}"
                    column_ranges[col_key] = (float(min_val), float(max_val))

    return column_ranges

@st.cache_data(ttl=Config.CACHE_TTL_SECONDS)
def get_table_info(_conn: duckdb.DuckDBPyConnection, data_dir: str) -> tuple[list[str], list[str], dict[str, list[str]], dict[str, str], dict[str, tuple[float, float]], MergeKeys, list[str], list[str], bool]:
    """
    Scans the data directory for CSV files and returns information about them.
    - Caches the result for 10 minutes to avoid repeatedly scanning the file system.
    - Returns tables, columns, data types, min/max ranges, merge keys, actions taken, session values, and empty state flag.
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Detect merge strategy first
    merge_keys = Config.get_merge_keys()

    # Prepare datasets if longitudinal
    actions_taken = []
    if merge_keys.is_longitudinal:
        success, actions_taken = Config.get_merge_strategy().prepare_datasets(data_dir, merge_keys)

    behavioral_tables: list[str] = []
    demographics_columns: list[str] = []
    behavioral_columns: dict[str, list[str]] = {}
    column_dtypes: dict[str, str] = {}
    column_ranges: dict[str, tuple[float, float]] = {}

    all_csv_files = scan_csv_files(data_dir)
    is_empty_state = not all_csv_files

    if is_empty_state:
        # Return empty state with default merge keys
        return [], [], {}, {}, {}, merge_keys, [], [], True

    for f in all_csv_files:
        table_name = f.replace('.csv', '')
        is_demo_table = (f == Config.DEMOGRAPHICS_FILE)

        if not is_demo_table:
            behavioral_tables.append(table_name)

        table_path = os.path.join(data_dir, f)

        # Validate file structure first
        if not validate_csv_structure(table_path, f, merge_keys):
            continue

        try:
            # Fast metadata extraction
            current_cols, file_dtypes = extract_column_metadata_fast(table_path, table_name, is_demo_table, merge_keys)
            column_dtypes.update(file_dtypes)

            if is_demo_table:
                # For demographics, we need the full column list
                df_sample = pd.read_csv(table_path, nrows=0)  # Just headers
                demographics_columns = df_sample.columns.tolist()

                # Validate required demographic columns
                if 'age' not in demographics_columns:
                    st.info("Note: Age column not found in demographics. Age filtering will be disabled.")
                if 'sex' not in demographics_columns:
                    st.info("Note: Sex column not found in demographics. Sex filtering will be disabled.")
            else:
                behavioral_columns[table_name] = current_cols

            # Fast range calculation
            file_ranges = calculate_numeric_ranges_fast(table_path, table_name, is_demo_table, file_dtypes, merge_keys)
            column_ranges.update(file_ranges)

        except pd.errors.EmptyDataError:
            st.error(f"Error: File '{f}' is empty.")
            continue
        except pd.errors.ParserError as e:
            st.error(f"Error parsing '{f}': {e}")
            continue
        except FileNotFoundError:
            st.error(f"Error: File '{f}' not found.")
            continue
        except PermissionError:
            st.error(f"Error: Permission denied reading '{f}'.")
            continue
        except Exception as e:
            st.error(f"Unexpected error reading '{f}': {e}")
            continue

    # Extract unique session values for longitudinal data
    session_values = get_unique_session_values(data_dir, merge_keys)

    return behavioral_tables, demographics_columns, behavioral_columns, column_dtypes, column_ranges, merge_keys, actions_taken, session_values, False

def generate_base_query_logic(demographic_filters: dict[str, Any], behavioral_filters: list[dict[str, Any]], tables_to_join: list[str], merge_keys: MergeKeys) -> tuple[str, list[Any]]:
    """
    Generates the common FROM, JOIN, and WHERE clauses for all queries.
    This centralized logic ensures consistency between the count and data queries.
    """
    if not tables_to_join:
        # If no tables are selected for joining, we can still filter the demo table
        tables_to_join = [Config.get_demographics_table_name()]

    # If session filtering is active, ensure we have behavioral tables to filter against
    # Since session_num is primarily in behavioral tables, we need at least one for filtering
    if ('sessions' in demographic_filters and demographic_filters['sessions']):
        # Check if we only have demographics table
        behavioral_tables_present = any(table != Config.get_demographics_table_name() for table in tables_to_join)

        if not behavioral_tables_present:
            # We need to add some behavioral tables for session filtering to work
            # Choose a table that's likely to have good session coverage
            # Prefer tables like ANT, demographics, or other core behavioral measures
            try:
                csv_files = scan_csv_files(Config.DATA_DIR)
                preferred_tables = ['ANT', 'demographics', 'DKEFS Trails', 'Dot Probe']

                # First try to find one of the preferred tables
                for preferred in preferred_tables:
                    for csv_file in csv_files:
                        table_name = csv_file.replace('.csv', '')
                        if (table_name == preferred and
                            table_name != Config.get_demographics_table_name()):
                            tables_to_join.append(table_name)
                            break
                    if len(tables_to_join) > 1:  # Found a table
                        break

                # If no preferred table found, use any behavioral table
                if len(tables_to_join) == 1:  # Still only demographics
                    for csv_file in csv_files:
                        table_name = csv_file.replace('.csv', '')
                        if table_name != Config.get_demographics_table_name():
                            tables_to_join.append(table_name)
                            break  # Just add one table to enable session filtering
            except Exception:
                pass  # If we can't find tables, proceed with demographics only

    # --- Build FROM and JOIN clauses ---
    base_table_path = os.path.join(Config.DATA_DIR, Config.DEMOGRAPHICS_FILE).replace('\\', '/')
    from_join_clause = f"FROM read_csv_auto('{base_table_path}') AS demo"
    # Create a set of all tables that need to be joined, including those for filtering
    all_join_tables: set[str] = set(tables_to_join)
    for bf in behavioral_filters:
        if bf['table']:
            all_join_tables.add(bf['table'])

    for table in all_join_tables:
        if table == Config.get_demographics_table_name():
            continue  # Skip joining demo to itself
        table_path = os.path.join(Config.DATA_DIR, f"{table}.csv").replace('\\', '/')
        merge_column = merge_keys.get_merge_column()
        from_join_clause += f"""
        LEFT JOIN read_csv_auto('{table_path}') AS {table}
        ON demo.{merge_column} = {table}.{merge_column}"""

    # --- Build WHERE clause ---
    where_clauses: list[str] = []
    params: dict[str, Any] = {}

    # 1. Demographic Filters
    if 'age_range' in demographic_filters and demographic_filters['age_range']:
        where_clauses.append("demo.age BETWEEN ? AND ?")
        params['age_min'] = demographic_filters['age_range'][0]
        params['age_max'] = demographic_filters['age_range'][1]

    if 'sex' in demographic_filters and demographic_filters['sex']:
        numeric_sex_values = [Config.SEX_MAPPING[s] for s in demographic_filters['sex']]
        placeholders = ', '.join(['?' for _ in numeric_sex_values])
        where_clauses.append(f"demo.sex IN ({placeholders})")
        for i, num_sex in enumerate(numeric_sex_values):
            params[f'sex_{i}'] = num_sex

    # RS1 Study Filters
    if 'studies' in demographic_filters and demographic_filters['studies']:
        study_conditions = []
        for study in demographic_filters['studies']:
            study_conditions.append(f"demo.{study} = ?")
            params[f'study_{study}'] = 1
        if study_conditions:
            where_clauses.append(f"({' OR '.join(study_conditions)})")

    # Rockland Sample1 Substudy Filters (string-based filtering on all_studies column)
    if 'substudies' in demographic_filters and demographic_filters['substudies']:
        substudy_conditions = []
        for substudy in demographic_filters['substudies']:
            substudy_conditions.append("demo.all_studies LIKE ?")
            params[f'substudy_{substudy}'] = f'%{substudy}%'
        if substudy_conditions:
            where_clauses.append(f"({' OR '.join(substudy_conditions)})")

    # Session Filters - check if any joined table has session_num column
    if 'sessions' in demographic_filters and demographic_filters['sessions']:
        # Build OR conditions for session filtering across all joined tables with session_num
        session_conditions = []
        session_placeholders = ', '.join(['?' for _ in demographic_filters['sessions']])

        for table in all_join_tables:
            if table != Config.get_demographics_table_name():  # Skip demo table
                table_alias = get_table_alias(table)
                session_conditions.append(f"{table_alias}.session_num IN ({session_placeholders})")

        if session_conditions:
            where_clauses.append(f"({' OR '.join(session_conditions)})")
            # Add session parameters - one set for each table condition
            for _ in session_conditions:
                for session in demographic_filters['sessions']:
                    params[f'session_{len(params)}'] = session

    # 2. Behavioral Filters
    for i, b_filter in enumerate(behavioral_filters):
        if b_filter['table'] and b_filter['column']:
            df_name = get_table_alias(b_filter['table'])
            col_name = f'"{b_filter["column"]}"'
            where_clauses.append(f"{df_name}.{col_name} BETWEEN ? AND ?")
            params[f"b_filter_min_{i}"] = b_filter['min_val']
            params[f"b_filter_max_{i}"] = b_filter['max_val']

    where_clause = ""
    if where_clauses:
        where_clause = "\nWHERE " + " AND ".join(where_clauses)

    return f"{from_join_clause}{where_clause}", list(params.values())

def generate_data_query(base_query_logic: str, params: list[Any], selected_tables: list[str], selected_columns: dict[str, list[str]]) -> tuple[Optional[str], Optional[list[Any]]]:
    """Generates the full SQL query to fetch data."""
    if not base_query_logic:
        return None, None

    select_clause = "SELECT demo.*"
    for table, columns in selected_columns.items():
        if table in selected_tables and columns:
            for col in columns:
                select_clause += f', {table}."{col}"'

    return f"{select_clause} {base_query_logic}", params

def generate_count_query(base_query_logic: str, params: list[Any], merge_keys: MergeKeys) -> tuple[Optional[str], Optional[list[Any]]]:
    """Generates a query to count distinct participants."""
    if not base_query_logic:
        return None, None
    merge_column = merge_keys.get_merge_column()
    select_clause = f"SELECT COUNT(DISTINCT demo.{merge_column})"
    return f"{select_clause} {base_query_logic}", params


# --- UI Functions ---

def render_file_upload_section(is_empty_state: bool = False) -> bool:
    """
    Render file upload interface.

    Args:
        is_empty_state: Whether this is being shown in an empty state (no existing data)

    Returns:
        bool: True if files were successfully uploaded and saved
    """
    if is_empty_state:
        st.header("🚀 Welcome to Lab Data Query Tool")
        st.markdown("""
        **Get started by uploading your CSV data files!**

        This tool helps you merge, filter, and export laboratory research data. To begin:
        1. Upload one or more CSV files below
        2. Include a demographics file with participant information
        3. Start exploring your data with filters and queries
        """)

    st.subheader("📁 Upload CSV Files")

    # Show format requirements - only use expander in empty state to avoid nesting
    if is_empty_state:
        with st.expander("📋 File Format Requirements", expanded=True):
            st.markdown("""
            **File Limits:**
            - Maximum file size: 50MB per file
            - Maximum 1000 columns per file
            - UTF-8 encoding recommended

            """)
    else:
        # Show requirements inline when inside another expander
        st.markdown("""
        **📋 File Format Requirements:**
        - Files must be in CSV format (.csv extension)
        - Include at least one demographics file with participant information
        - Recommended to have a subject ID column (e.g., `ursi`, `subject_id`, `participant_id`)
        - For longitudinal data: include session identifier (e.g., `session_num`, `timepoint`)
        - Maximum file size: 50MB per file, UTF-8 encoding recommended
        """)

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose CSV files" if not is_empty_state else "Upload your CSV data files to get started",
        accept_multiple_files=True,
        type=['csv'],
        help="Select one or more CSV files. Drag and drop is supported!",
        key="file_uploader"
    )

    files_uploaded = False

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected for upload:**")

        # Validate all files first
        all_valid = True
        file_summaries = []

        for uploaded_file in uploaded_files:
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Basic file info
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    st.write(f"📄 **{uploaded_file.name}** ({file_size_mb:.1f} MB)")

                # Validate file
                errors, df = validate_csv_file(uploaded_file)

                if errors:
                    with col1:
                        for error in errors:
                            st.error(f"❌ {error}")
                    all_valid = False
                else:
                    with col1:
                        st.success(f"✅ Valid CSV: {len(df)} rows, {len(df.columns)} columns")

                        # Show column preview
                        if len(df.columns) <= 20:
                            st.caption(f"Columns: {', '.join(df.columns)}")
                        else:
                            st.caption(f"Columns: {', '.join(df.columns[:15])}... (+{len(df.columns) - 15} more)")

                    file_summaries.append({
                        'name': uploaded_file.name,
                        'rows': len(df),
                        'columns': list(df.columns),
                        'file_obj': uploaded_file,
                        'dataframe': df
                    })

        st.markdown("---")

        # Show overall validation summary
        if all_valid and file_summaries:
            st.success(f"🎉 All {len(file_summaries)} files passed validation!")

            # Show merge compatibility check
            has_demographics = any('demographics' in f['name'].lower() for f in file_summaries)
            has_id_columns = any(
                any(col.lower() in ['ursi', 'subject_id', 'participant_id', 'customid'] for col in f['columns'])
                for f in file_summaries
            )

            if has_demographics:
                st.info("✅ Demographics file detected")
            else:
                st.warning("⚠️ No demographics file detected - consider including a demographics.csv file")

            if has_id_columns:
                st.info("✅ Subject ID columns detected")
            else:
                st.warning("⚠️ No standard ID columns detected - you may need to configure column names")

            # Save button
            if st.button("💾 Save Files to Data Directory", type="primary"):
                with st.spinner("Saving files..."):
                    saved_files = save_uploaded_files_to_data_dir(
                        [f['file_obj'] for f in file_summaries],
                        Config.DATA_DIR
                    )

                    if saved_files:
                        st.success(f"🎉 Successfully saved {len(saved_files)} files!")
                        st.info("🔄 Refreshing application to load your new data...")

                        # Clear cache to refresh table info
                        get_table_info.clear()
                        files_uploaded = True

                        # Small delay to let users see the success message
                        time.sleep(1)
                        st.rerun()

        elif not all_valid:
            st.error("❌ Please fix the validation errors above before uploading")

    elif is_empty_state:
        st.info("👆 Upload your CSV files using the file uploader above to get started!")

    return files_uploaded

def render_empty_state_welcome():
    """Render a welcome screen for users with no data."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>🔬 The Basic Scientist's Data Query and Merge Tool</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Upload your CSV data files to start exploring and merging research datasets
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### Getting Started
        This screen is to help you get started by uploading your CSV files.
        **To begin:** Upload your CSV files using the section below.

        **File Format Requirements:**
        - Files must be in CSV format (.csv extension)
        - Include at least one demographics file with participant information (e.g., `demographics.csv`)
        - Recommended to have a subject ID column (e.g., `ursi`, `subject_id`, `participant_id`)
        - For longitudinal data: include session identifier (e.g., `session_num`, `timepoint`)
        - Maximum file size: 50MB per file, UTF-8 encoding recommended.

        **Example Demographics Column Structures:**
        - Cross-sectional: `ursi, age, sex, ...`
        - Longitudinal: `ursi, session_num, age, sex, ...`
        - Rockland Sample1 (multi-session, multi-study): `ursi, session_num, all_studies, age, sex, ...`
    """)

def sync_table_order() -> None:
    """Callback to move newly selected tables to the bottom of the list."""
    current_selection: list[str] = st.session_state.multiselect_key
    old_order: list[str] = st.session_state.get('table_order', [])
    added = [item for item in current_selection if item not in old_order]
    new_order = [item for item in old_order if item in current_selection] + added
    st.session_state.table_order = new_order

def add_behavioral_filter() -> None:
    """Adds a new blank filter dictionary to the session state."""
    st.session_state.behavioral_filters.append({
        'id': time.time(),
        'table': None,
        'column': None,
        'min_val': Config.DEFAULT_FILTER_RANGE[0],
        'max_val': Config.DEFAULT_FILTER_RANGE[1]
    })

def remove_behavioral_filter(filter_id: float) -> None:
    """Removes a filter from session state based on its unique ID."""
    st.session_state.behavioral_filters = [
        f for f in st.session_state.behavioral_filters if f['id'] != filter_id
    ]

def render_demographic_filters(demographics_columns: list[str], merge_keys: MergeKeys, session_values: list[str]) -> tuple[Optional[tuple[int, int]], list[str], list[str], list[str], list[str]]:
    """Renders demographic filter UI and returns filter values."""
    st.subheader("Demographic Filters")

    # RS1 Study Selection (if in RS1 format)
    selected_studies = []
    is_rs1 = detect_rs1_format(demographics_columns)
    if is_rs1:
        st.subheader("Study Selection")
        cols = st.columns(len(Config.RS1_STUDY_COLUMNS))
        for i, study_col in enumerate(Config.RS1_STUDY_COLUMNS):
            with cols[i]:
                if st.checkbox(Config.RS1_STUDY_LABELS[study_col], value=True, key=f"study_{study_col}"):
                    selected_studies.append(study_col)

        # Session Selection dropdown
        st.subheader("Session Selection")
        selected_sessions = st.multiselect(
            "Select Sessions",
            options=Config.SESSION_OPTIONS,
            default=Config.DEFAULT_SESSION_SELECTION,
            key="session_selection"
        )
        st.markdown("---")

    # Rockland Sample1 Substudy Selection (if in Rockland format)
    selected_substudies = []
    is_rockland = detect_rockland_format(demographics_columns)
    if is_rockland:
        st.subheader("Substudy Selection")
        selected_substudies = st.multiselect(
            "Select Base Studies",
            options=Config.ROCKLAND_BASE_STUDIES,
            default=Config.DEFAULT_ROCKLAND_STUDIES,
            key="rockland_substudy_selection",
            help="Select which base studies to include in the dataset"
        )
        st.markdown("---")

    if merge_keys.is_longitudinal and session_values:
        # Dynamic session selection for longitudinal data
        st.subheader("Session Selection")
        selected_sessions = st.multiselect(
            f"Select {merge_keys.session_id} Values",
            options=session_values,
            default=session_values,  # Default to all sessions
            key="session_selection"
        )
        st.markdown("---")
    else:
        selected_sessions = []


    age_range = None
    if 'age' in demographics_columns:
        age_range = st.slider(
            "Select Age Range",
            *Config.DEFAULT_AGE_RANGE,
            Config.DEFAULT_AGE_SELECTION
        )

    selected_sex = []
    if 'sex' in demographics_columns:
        selected_sex = st.multiselect(
            "Select Sex",
            Config.SEX_OPTIONS,
            Config.DEFAULT_SEX_SELECTION
        )

    return age_range, selected_sex, selected_studies, selected_sessions, selected_substudies

def render_behavioral_filters(all_filterable_tables: list[str], demographics_columns: list[str],
                            behavioral_columns_by_table: dict[str, list[str]],
                            column_dtypes: dict[str, str], column_ranges: dict[str, tuple[float, float]]) -> None:
    """Renders phenotypic filter UI."""
    st.subheader("Phenotypic Filters")

    for i, behavioral_filter in enumerate(st.session_state.behavioral_filters):
        with st.container():
            st.markdown(f"**Filter {i+1}**")
            behavioral_filter['table'] = st.selectbox(
                "Table",
                options=all_filterable_tables,
                index=None,
                key=f"behavioral_filter_table_{behavioral_filter['id']}"
            )

            numeric_cols = []
            if behavioral_filter['table']:
                df_name = get_table_alias(behavioral_filter['table'])
                all_cols_for_table = demographics_columns if df_name == 'demo' else behavioral_columns_by_table.get(behavioral_filter['table'], [])

                numeric_cols = [
                    col for col in all_cols_for_table
                    if is_numeric_column(column_dtypes.get(f"{df_name}.{col}", ''))
                ]

            behavioral_filter['column'] = st.selectbox(
                "Column (Numeric Only)",
                options=numeric_cols,
                index=None,
                key=f"behavioral_filter_col_{behavioral_filter['id']}"
            )

            if behavioral_filter['column']:
                df_name = get_table_alias(behavioral_filter['table'])
                min_val, max_val = column_ranges.get(
                    f"{df_name}.{behavioral_filter['column']}",
                    Config.DEFAULT_FILTER_RANGE
                )

                behavioral_filter['min_val'], behavioral_filter['max_val'] = st.slider(
                   f"Range for {behavioral_filter['column']}",
                   min_value=min_val,
                   max_value=max_val,
                   value=(min_val, max_val),
                   key=f"behavioral_filter_range_{behavioral_filter['id']}"
                )

            st.button(
                "Remove",
                key=f"remove_behavioral_filter_{behavioral_filter['id']}",
                on_click=remove_behavioral_filter,
                args=(behavioral_filter['id'],)
            )
            st.markdown("---")

    st.button("Add Phenotypic Filter", on_click=add_behavioral_filter)

    # Information box explaining phenotypic filters
    st.info("💡 Filter by phenotypic variables, e.g. flanker accuracy > 0.75; MOCA > 25")

def render_table_selection(available_tables: list[str], behavioral_columns_by_table: dict[str, list[str]]) -> dict[str, list[str]]:
    """Renders table and column selection UI and returns selected columns."""
    st.header("2. Select Data for Export")

    st.multiselect(
        "Choose tables to merge:",
        options=available_tables,
        key='multiselect_key',
        on_change=sync_table_order
    )

    tables_in_use = st.session_state.get('table_order', [])
    selected_columns_per_table = {}

    if not tables_in_use:
        st.info("Select one or more tables above to choose which columns to export.")
    else:
        for table in tables_in_use:
            with st.expander(f"Columns for '{table}'", expanded=True):
                all_cols = behavioral_columns_by_table.get(table, [])
                selected_columns_per_table[table] = st.multiselect(
                    f"Select columns from {table}",
                    options=all_cols,
                    default=[],
                    key=f"cols_{table}"
                )

    return selected_columns_per_table

def enwiden_longitudinal_data(df: pd.DataFrame, merge_keys: MergeKeys, selected_columns_per_table: dict[str, list[str]]) -> pd.DataFrame:
    """
    Pivots longitudinal data so each subject has one row with session-specific columns.
    Transforms columns like 'age' into 'age_BAS1', 'age_BAS2', etc.
    """
    if not merge_keys.is_longitudinal or merge_keys.session_id not in df.columns:
        return df

    # Identify columns to pivot (exclude ID columns)
    id_columns = {merge_keys.primary_id}
    if merge_keys.composite_id and merge_keys.composite_id in df.columns:
        id_columns.add(merge_keys.composite_id)
    if merge_keys.session_id in df.columns:
        id_columns.add(merge_keys.session_id)

    # Get all columns that should be pivoted
    pivot_columns = [col for col in df.columns if col not in id_columns]

    # Keep static columns (those that don't vary by session) separate
    # These are typically demographic variables that are consistent across sessions
    static_columns = []
    for col in pivot_columns:
        # Check if this column has the same value across all sessions for each subject
        if col in df.columns:
            # Group by primary ID and check if the column values are consistent across sessions
            grouped = df.groupby(merge_keys.primary_id)[col].nunique()
            if (grouped <= 1).all():  # All subjects have at most 1 unique value across sessions
                static_columns.append(col)

    # Separate static and dynamic columns
    dynamic_columns = [col for col in pivot_columns if col not in static_columns]

    # Start with static data (one row per subject)
    if static_columns:
        static_df = df.groupby(merge_keys.primary_id)[static_columns].first().reset_index()
    else:
        static_df = df[[merge_keys.primary_id]].drop_duplicates().reset_index(drop=True)

    # Pivot dynamic columns
    if dynamic_columns:
        # Create pivot table for each dynamic column
        pivoted_dfs = []

        for col in dynamic_columns:
            pivot_df = df.pivot_table(
                index=merge_keys.primary_id,
                columns=merge_keys.session_id,
                values=col,
                aggfunc='first'  # Take first value if duplicates exist
            )

            # Flatten column names: 'age_BAS1', 'age_BAS2', etc.
            pivot_df.columns = [f"{col}_{session}" for session in pivot_df.columns]
            pivot_df = pivot_df.reset_index()

            pivoted_dfs.append(pivot_df)

        # Merge all pivoted columns
        widened_df = static_df
        for pivot_df in pivoted_dfs:
            widened_df = widened_df.merge(pivot_df, on=merge_keys.primary_id, how='left')
    else:
        widened_df = static_df

    return widened_df

def render_results_section(base_query_for_count: str, params_for_count: list[Any],
                         tables_in_use: list[str], selected_columns_per_table: dict[str, list[str]],
                         age_range: Optional[tuple[int, int]], selected_sex: list[str], selected_studies: list[str],
                         selected_sessions: list[str], selected_substudies: list[str], con: duckdb.DuckDBPyConnection, merge_keys: MergeKeys) -> None:
    """Renders the data generation and download section."""
    st.header("3. Generate & Download Data")

    # Add enwiden option for longitudinal data
    enwiden_data = False
    if merge_keys.is_longitudinal:
        enwiden_data = st.checkbox(
            "Enwiden by session",
            value=False,
            help="Pivot data so each subject has one row with session-specific columns (e.g., age_BAS1, age_BAS2)"
        )

    run_query_button = st.button("Generate Merged Data", type="primary", disabled=not tables_in_use)

    if run_query_button:
        data_query, data_params = generate_data_query(
            base_query_for_count, params_for_count, tables_in_use, selected_columns_per_table
        )

        if data_query:
            try:
                with st.spinner("🔄 Running query and merging data..."):
                    start_time = time.time()
                    result_df = con.execute(data_query, data_params).fetchdf()

                    # Apply enwiden transformation if requested for longitudinal data
                    if enwiden_data and merge_keys.is_longitudinal:
                        with st.spinner("🔄 Enwidening data by session..."):
                            result_df = enwiden_longitudinal_data(result_df, merge_keys, selected_columns_per_table)

                    end_time = time.time()

                # Display results info
                transform_info = ""
                if enwiden_data and merge_keys.is_longitudinal:
                    original_rows = con.execute(data_query, data_params).fetchdf().shape[0]
                    transform_info = f" (enwidened from {original_rows} session-rows to {len(result_df)} subject-rows)"

                st.success(f"✅ Query completed in {end_time - start_time:.2f} seconds.")
                st.subheader("Query Results")
                st.write(f"**Total matching participants found:** {len(result_df)}{transform_info}")
                st.dataframe(result_df.head(Config.MAX_DISPLAY_ROWS))

                csv = result_df.to_csv(index=False).encode('utf-8')
                filename_parts = [
                    'data',
                    f"age{age_range[0]}-{age_range[1]}" if age_range else '',
                    '_'.join(selected_sex).lower() if selected_sex else '',
                    '_'.join([s.replace('is_', '') for s in selected_studies]) if selected_studies else '',
                    '_'.join(selected_sessions).lower() if selected_sessions else '',
                    '_'.join(selected_substudies).lower() if selected_substudies else '',
                    '_'.join(tables_in_use),
                    'enwidened' if enwiden_data and merge_keys.is_longitudinal else ''
                ]
                suggested_filename = '_'.join(filter(None, filename_parts)) + '.csv'
                st.download_button(
                    "📥 Download Full Dataset as CSV",
                    csv,
                    suggested_filename,
                    'text/csv'
                )

            except Exception as e:
                st.error(f"An error occurred during the database query: {e}")
                st.code(data_query)
        else:
            st.warning("Please select at least one behavioral data table to export.")

# --- Main Application ---
def main() -> None:
    """Main application entry point."""
    # Parse CLI arguments first to configure the app
    Config.parse_cli_args()

    # Handle empty state vs normal application flow
    con = get_db_connection()
    available_tables, demographics_columns, behavioral_columns_by_table, column_dtypes, column_ranges, merge_keys, actions_taken, session_values, is_empty_state = get_table_info(con, Config.DATA_DIR)

    if is_empty_state:
        # Show empty state interface
        render_empty_state_welcome()
        st.markdown("---")
        render_file_upload_section(is_empty_state=True)
        return  # Exit early - don't show the main interface

    # Normal application flow with data
    st.title("🔬 The Basic Scientist's Data Query and Merge Tool")

    # Initialize session state
    if 'table_order' not in st.session_state:
        st.session_state.table_order = []
    if 'behavioral_filters' not in st.session_state:
        st.session_state.behavioral_filters = []

    # File upload section (for adding more data)
    with st.expander("📁 Upload Additional Files", expanded=False):
        render_file_upload_section(is_empty_state=False)

    # Display merge strategy info
    if merge_keys.is_longitudinal:
        st.info(f"📊 **Longitudinal data detected** - Using `{merge_keys.primary_id}` + `{merge_keys.session_id}` → `{merge_keys.composite_id}`")
        if actions_taken:
            with st.expander("📋 Dataset Preparation Actions", expanded=False):
                for action in actions_taken:
                    st.text(action)
    else:
        st.info(f"📊 **Cross-sectional data detected** - Using `{merge_keys.primary_id}` for merging")

    # Live participant count placeholder
    st.subheader("Live Participant Count")
    count_placeholder = st.empty()
    st.markdown("---")

    # Main UI layout
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("1. Define Cohort Filters")

        # Demographic filters
        age_range, selected_sex, selected_studies, selected_sessions, selected_substudies = render_demographic_filters(demographics_columns, merge_keys, session_values)

        st.markdown("---")

        # Phenotypic filters
        all_filterable_tables = [Config.get_demographics_table_name()] + available_tables
        render_behavioral_filters(
            all_filterable_tables, demographics_columns,
            behavioral_columns_by_table, column_dtypes, column_ranges
        )

        # Calculate live count
        demographic_filters_state = {'age_range': age_range, 'sex': selected_sex, 'studies': selected_studies, 'sessions': selected_sessions, 'substudies': selected_substudies}
        active_behavioral_filters = [
            filter_config for filter_config in st.session_state.behavioral_filters
            if filter_config['table'] and filter_config['column']
        ]
        tables_for_count = set(st.session_state.get('table_order', []))
        for filter_config in active_behavioral_filters:
            tables_for_count.add(filter_config['table'])

        base_query_for_count, params_for_count = generate_base_query_logic(
            demographic_filters_state, active_behavioral_filters, list(tables_for_count), merge_keys
        )
        count_query, count_params = generate_count_query(base_query_for_count, params_for_count, merge_keys)

        if count_query:
            # print(count_query, count_params)  ## DEBUG
            count_result = con.execute(count_query, count_params).fetchone()[0]
            count_placeholder.metric("Matching Participants", count_result)
        else:
            count_placeholder.metric("Matching Participants", "N/A")

    with col2:
        # Table and column selection
        selected_columns_per_table = render_table_selection(available_tables, behavioral_columns_by_table)
        tables_in_use = st.session_state.get('table_order', [])

    # Results section
    st.markdown("---")
    render_results_section(
        base_query_for_count, params_for_count, tables_in_use,
        selected_columns_per_table, age_range, selected_sex, selected_studies, selected_sessions, selected_substudies, con, merge_keys
    )

if __name__ == "__main__":
    main()
