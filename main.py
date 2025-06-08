import streamlit as st
import duckdb
import pandas as pd
import os
import time
import numpy as np
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Any, Set

# --- Configuration ---
class Config:
    # File and directory settings
    DATA_DIR = 'data'
    DEMOGRAPHICS_FILE = 'demographics.csv'
    PARTICIPANT_ID_COLUMN = 'customID'
    
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
    
    # Session selection options
    SESSION_OPTIONS = ['BAS1', 'BAS2', 'BAS3', 'FLU1', 'FLU2', 'FLU3', 'NFB', 'TRT', 'TRT2']
    DEFAULT_SESSION_SELECTION = ['BAS1', 'BAS2', 'BAS3', 'FLU1', 'FLU2', 'FLU3', 'NFB', 'TRT', 'TRT2']
    
    @classmethod
    def get_demographics_table_name(cls) -> str:
        return cls.DEMOGRAPHICS_FILE.replace('.csv', '')
    
    @classmethod
    def parse_cli_args(cls) -> None:
        """Parse command line arguments and update Config class attributes."""
        parser = argparse.ArgumentParser(
            description='Lab Data Query and Merge Tool - Configure runtime parameters',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
            help='Column name for participant ID across all tables'
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
                cls.MAX_DISPLAY_ROWS = args.max_display_rows
                cls.CACHE_TTL_SECONDS = args.cache_ttl_seconds
                cls.DEFAULT_AGE_SELECTION = (args.default_age_min, args.default_age_max)
            except SystemExit:
                # Handle --help or invalid arguments gracefully
                pass


# --- Page Setup ---
st.set_page_config(
    page_title="DuckDB Lab Query Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Helper Functions ---

@st.cache_resource
def get_db_connection() -> duckdb.DuckDBPyConnection:
    """
    Establishes and caches a connection to an in-memory DuckDB database.
    """
    return duckdb.connect(database=':memory:', read_only=False)

def scan_csv_files(data_dir: str) -> List[str]:
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

def detect_rs1_format(demographics_columns: List[str]) -> bool:
    """
    Detects if the demographics file is in RS1 format by checking for study columns.
    """
    return all(col in demographics_columns for col in Config.RS1_STUDY_COLUMNS)

def validate_csv_structure(file_path: str, filename: str) -> bool:
    """
    Validates basic CSV structure and required columns.
    """
    try:
        # Check if file can be read and has the required participant ID column
        df_headers = pd.read_csv(file_path, nrows=0)
        
        if Config.PARTICIPANT_ID_COLUMN not in df_headers.columns:
            st.warning(f"Warning: '{filename}' missing required column '{Config.PARTICIPANT_ID_COLUMN}'.")
            return False
            
        if len(df_headers.columns) == 0:
            st.warning(f"Warning: '{filename}' has no columns.")
            return False
            
        return True
        
    except Exception as e:
        st.error(f"Error validating '{filename}': {e}")
        return False

def extract_column_metadata_fast(file_path: str, table_name: str, is_demo_table: bool) -> Tuple[List[str], Dict[str, str]]:
    """
    Extracts column information and data types from CSV without loading full data.
    """
    df_name = get_table_alias(table_name if not is_demo_table else Config.get_demographics_table_name())
    
    # Read only first few rows for metadata
    df_sample = pd.read_csv(file_path, nrows=100)
    columns = [col for col in df_sample.columns if col != Config.PARTICIPANT_ID_COLUMN]
    
    column_dtypes = {}
    for col in df_sample.columns:
        if col == Config.PARTICIPANT_ID_COLUMN:
            continue
        col_key = f"{df_name}.{col}"
        column_dtypes[col_key] = str(df_sample[col].dtype)
    
    return columns, column_dtypes

def calculate_numeric_ranges_fast(file_path: str, table_name: str, is_demo_table: bool, column_dtypes: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    """
    Calculates min/max ranges for numeric columns using chunked reading.
    """
    df_name = get_table_alias(table_name if not is_demo_table else Config.get_demographics_table_name())
    column_ranges = {}
    
    # Get numeric columns from dtypes
    numeric_cols = []
    for col_key, dtype_str in column_dtypes.items():
        if col_key.startswith(f"{df_name}.") and is_numeric_column(dtype_str):
            col_name = col_key.split('.', 1)[1]
            if col_name != Config.PARTICIPANT_ID_COLUMN:
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
def get_table_info(_conn: duckdb.DuckDBPyConnection, data_dir: str) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, str], Dict[str, Tuple[float, float]]]:
    """
    Scans the data directory for CSV files and returns information about them.
    - Caches the result for 10 minutes to avoid repeatedly scanning the file system.
    - Returns tables, columns, data types, and min/max ranges for numeric columns.
    """
    behavioral_tables: List[str] = []
    demographics_columns: List[str] = []
    behavioral_columns: Dict[str, List[str]] = {}
    column_dtypes: Dict[str, str] = {}
    column_ranges: Dict[str, Tuple[float, float]] = {}

    all_csv_files = scan_csv_files(data_dir)
    if not all_csv_files:
        return [], [], {}, {}, {}

    for f in all_csv_files:
        table_name = f.replace('.csv', '')
        is_demo_table = (f == Config.DEMOGRAPHICS_FILE)
        
        if not is_demo_table:
            behavioral_tables.append(table_name)

        table_path = os.path.join(data_dir, f)
        
        # Validate file structure first
        if not validate_csv_structure(table_path, f):
            continue
            
        try:
            # Fast metadata extraction
            current_cols, file_dtypes = extract_column_metadata_fast(table_path, table_name, is_demo_table)
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
            file_ranges = calculate_numeric_ranges_fast(table_path, table_name, is_demo_table, file_dtypes)
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

    return behavioral_tables, demographics_columns, behavioral_columns, column_dtypes, column_ranges

def generate_base_query_logic(demographic_filters: Dict[str, Any], behavioral_filters: List[Dict[str, Any]], tables_to_join: List[str]) -> Tuple[str, List[Any]]:
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
    all_join_tables: Set[str] = set(tables_to_join)
    for bf in behavioral_filters:
        if bf['table']:
            all_join_tables.add(bf['table'])
            
    for table in all_join_tables:
        if table == Config.get_demographics_table_name(): 
            continue  # Skip joining demo to itself
        table_path = os.path.join(Config.DATA_DIR, f"{table}.csv").replace('\\', '/')
        from_join_clause += f"""
        LEFT JOIN read_csv_auto('{table_path}') AS {table}
        ON demo.{Config.PARTICIPANT_ID_COLUMN} = {table}.{Config.PARTICIPANT_ID_COLUMN}"""

    # --- Build WHERE clause ---
    where_clauses: List[str] = []
    params: Dict[str, Any] = {}
    
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

def generate_data_query(base_query_logic: str, params: List[Any], selected_tables: List[str], selected_columns: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[List[Any]]]:
    """Generates the full SQL query to fetch data."""
    if not base_query_logic:
        return None, None
        
    select_clause = "SELECT demo.*"
    for table, columns in selected_columns.items():
        if table in selected_tables and columns:
            for col in columns:
                select_clause += f', {table}."{col}"'
    
    return f"{select_clause} {base_query_logic}", params

def generate_count_query(base_query_logic: str, params: List[Any]) -> Tuple[Optional[str], Optional[List[Any]]]:
    """Generates a query to count distinct participants."""
    if not base_query_logic:
        return None, None
    select_clause = f"SELECT COUNT(DISTINCT demo.{Config.PARTICIPANT_ID_COLUMN})"
    return f"{select_clause} {base_query_logic}", params


# --- UI Functions ---
def sync_table_order() -> None:
    """Callback to move newly selected tables to the bottom of the list."""
    current_selection: List[str] = st.session_state.multiselect_key
    old_order: List[str] = st.session_state.get('table_order', [])
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

def render_demographic_filters(demographics_columns: List[str]) -> Tuple[Optional[Tuple[int, int]], List[str], List[str], List[str]]:
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
    
    return age_range, selected_sex, selected_studies, selected_sessions

def render_behavioral_filters(all_filterable_tables: List[str], demographics_columns: List[str], 
                            behavioral_columns_by_table: Dict[str, List[str]], 
                            column_dtypes: Dict[str, str], column_ranges: Dict[str, Tuple[float, float]]) -> None:
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

def render_table_selection(available_tables: List[str], behavioral_columns_by_table: Dict[str, List[str]]) -> Dict[str, List[str]]:
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

def render_results_section(base_query_for_count: str, params_for_count: List[Any], 
                         tables_in_use: List[str], selected_columns_per_table: Dict[str, List[str]], 
                         age_range: Optional[Tuple[int, int]], selected_sex: List[str], selected_studies: List[str],
                         selected_sessions: List[str], con: duckdb.DuckDBPyConnection) -> None:
    """Renders the data generation and download section."""
    st.header("3. Generate & Download Data")
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
                    end_time = time.time()

                st.success(f"✅ Query completed in {end_time - start_time:.2f} seconds.")
                st.subheader("Query Results")
                st.write(f"**Total matching participants found:** {len(result_df)}")
                st.dataframe(result_df.head(Config.MAX_DISPLAY_ROWS))

                csv = result_df.to_csv(index=False).encode('utf-8')
                filename_parts = [
                    'data', 
                    f"age{age_range[0]}-{age_range[1]}" if age_range else '', 
                    '_'.join(selected_sex).lower() if selected_sex else '', 
                    '_'.join([s.replace('is_', '') for s in selected_studies]) if selected_studies else '',
                    '_'.join(selected_sessions).lower() if selected_sessions else '',
                    '_'.join(tables_in_use)
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
    
    st.title("🔬 Lab Data Query and Merge Tool")

    # Initialize session state
    if 'table_order' not in st.session_state:
        st.session_state.table_order = []
    if 'behavioral_filters' not in st.session_state:
        st.session_state.behavioral_filters = []

    # Initialize database and data info
    con = get_db_connection()
    available_tables, demographics_columns, behavioral_columns_by_table, column_dtypes, column_ranges = get_table_info(con, Config.DATA_DIR)

    # Live participant count placeholder
    st.subheader("Live Participant Count")
    count_placeholder = st.empty()
    st.markdown("---")

    # Main UI layout
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("1. Define Cohort Filters")
        
        # Demographic filters
        age_range, selected_sex, selected_studies, selected_sessions = render_demographic_filters(demographics_columns)
        
        st.markdown("---")
        
        # Phenotypic filters
        all_filterable_tables = [Config.get_demographics_table_name()] + available_tables
        render_behavioral_filters(
            all_filterable_tables, demographics_columns, 
            behavioral_columns_by_table, column_dtypes, column_ranges
        )

        # Calculate live count
        demographic_filters_state = {'age_range': age_range, 'sex': selected_sex, 'studies': selected_studies, 'sessions': selected_sessions}
        active_behavioral_filters = [
            filter_config for filter_config in st.session_state.behavioral_filters 
            if filter_config['table'] and filter_config['column']
        ]
        tables_for_count = set(st.session_state.get('table_order', []))
        for filter_config in active_behavioral_filters: 
            tables_for_count.add(filter_config['table'])

        base_query_for_count, params_for_count = generate_base_query_logic(
            demographic_filters_state, active_behavioral_filters, list(tables_for_count)
        )
        count_query, count_params = generate_count_query(base_query_for_count, params_for_count)

        if count_query:
            print(count_query, count_params)  ## DEBUG
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
        selected_columns_per_table, age_range, selected_sex, selected_studies, selected_sessions, con
    )

if __name__ == "__main__":
    main()
