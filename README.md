	# The Basic Scientist's Data Query and Merge Tool

A Streamlit-based web application for laboratory research data filter, query, and merge. This tool allows researchers to interactively query, merge, and download CSV datasets using an intuitive interface - similar to LORIS or RedCap - and backed by DuckDB for efficient data processing.

## Who Wants/Needs This Application?

* You have data stored in multiple CSV files. 
* You'd like to be able to:
	* filter out some participants
	* select a subset of variables from across your CSVs
	* merge those vriables into a wide-format CSV for further analysis. 
* You are smart enough to avoid copy/paste operations in Excel.
* You want the power and efficiency of SQL, but don't want to **create a SQL database / SQL server**. 

**Key advantages:**
- **No dependence on your IT department** - this lightweight application runs on your laptop
- **Easy data updates** - to update your database, just update your CSV(s), drop them in your application's data folder, and restart the application
- **No database administration** - no need to negotiate with your IT department or cajole a tech-savvy colleague to update your SQL database
- **Familiar workflow** - works like web-based research databases you already know, but with complete local control

## Features

- **Smart Data Structure Detection**: Automatically detects cross-sectional vs longitudinal data formats
- **Flexible Column Configuration**: Works with any column naming convention via CLI parameters
- **Interactive Data Filtering**: Apply demographic and phenotypic filters to identify participant cohorts
- **Real-time Participant Count**: See matching participant counts update as you adjust filters
- **Intelligent Table Merging**: Adaptive merging strategy for different research data formats
- **Flexible Column Selection**: Choose specific columns from each table for export
- **Data Pivoting (Enwiden)**: Transform longitudinal data from long to wide format (e.g., `age_BAS1`, `age_BAS2`)
- **Fast Performance**: Optimized data loading and query execution using DuckDB
- **Export Functionality**: Download filtered and merged datasets as CSV files
- **RS1 Study Support**: Built-in support for multi-study datasets with session filtering
- **Legacy Compatibility**: Seamless support for existing `customID`-based datasets
- **Synthetic Data Generation**: Generate test datasets for development and training

## Getting Started

### Prerequisites

- Python 3.11 or higher
- UV package manager (recommended) or pip

### Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/SympatiCog/labdata_fqm.git
   ```

2. **Install dependencies:**
   ```bash
   cd labdata_fqm
   
   # Using UV (recommended; see https://docs.astral.sh/uv/getting-started/installation/)
   uv sync
   source .venv/bin/activate
   
   # Or in an existing python 3.11+ installation, you can use pip
   pip install -r requirements.txt
   ```

3. **Test with synthetic data (optional)**
   
   To explore the interface without using real data:
   ```bash
   python generate_synthetic_data.py
   streamlit run main.py -- --data-dir synthetic_data
   ```

### Quick Start

```bash
# Start the application with your data
streamlit run main.py

# Or with custom data directory
streamlit run main.py -- --data-dir your_data_folder
```

## 🚀 What's New: Flexible Data Structure Support

This application now **automatically adapts to your existing data** without requiring any file modifications:

- **Auto-Detection**: Instantly recognizes cross-sectional vs longitudinal data formats
- **Universal Column Support**: Works with any column naming convention (participant_id, SubjectID, ursi, etc.)
- **Smart Merging**: Adapts merge strategy based on your data structure
- **Legacy Compatibility**: Existing customID-based datasets work seamlessly
- **Zero Configuration**: Just point to your data folder and run - no setup required!

**Example**: Your data uses `participant_id` and `visit`? No problem:
```bash
streamlit run main.py -- --primary-id-column participant_id --session-column visit
```

## Data Setup

The application automatically adapts to your data structure - no file modifications required!

### Supported Data Formats

**Cross-sectional Data**: One row per participant
```csv
participant_id,age,sex,score
SUB001,25,1,85
SUB002,30,2,90
```

**Longitudinal Data**: Multiple rows per participant across sessions
```csv
participant_id,session,age,sex,score
SUB001,baseline,25,1,85
SUB001,followup,25,1,88
SUB002,baseline,30,2,90
SUB002,followup,30,2,92
```

**Legacy Format**: Existing `customID` datasets (fully supported)
```csv
customID,age,sex,score
SUB001_baseline,25,1,85
SUB001_followup,25,1,88
```

### Quick Setup

1. **Place your CSV files** in the `data/` directory (or specify custom directory)
2. **Include a demographics file** as the primary table (default: `demographics.csv`)
3. **Run the application** - it will auto-detect your column structure!

```bash
# Auto-detect everything (recommended)
streamlit run main.py

# Or specify your column names explicitly
streamlit run main.py -- --primary-id-column participant_id --session-column timepoint
```

### Example Data Structure

```
data/
├── demographics.csv      # Primary table (required)
├── VO2max.csv           # Phenotypic data table
├── woodcock_johnson.csv # Cognitive data table
├── flanker.csv          # Behavioral data table
└── ...                  # Additional data tables
```

### Column Name Flexibility

The application works with **any column naming convention**:

| Research Field | Primary ID | Session | CLI Example |
|----------------|------------|---------|-------------|
| Psychology | `participant_id` | `session` | `--primary-id-column participant_id --session-column session` |
| Clinical Trials | `SubjectID` | `Visit` | `--primary-id-column SubjectID --session-column Visit` |
| Neuroscience | `ursi` | `session_num` | Default settings |
| Epidemiology | `study_id` | `timepoint` | `--primary-id-column study_id --session-column timepoint` |

### Data Requirements

- **Required**: Common identifier column across all CSV files
- **Demographics table**: Primary table for joins (configurable name)
- **Optional**: `age`, `sex` columns in demographics for filtering
- **Automatic detection**: Longitudinal vs cross-sectional format
- **RS1 format support**: Auto-detected multi-study datasets

## Usage

### Running the Application

```bash
# Start the Streamlit application
streamlit run main.py

# Start Jupyter Lab for data analysis
jupyter lab
```

### Command Line Configuration

The application supports extensive runtime configuration via command line parameters. Use the double dash (`--`) separator when running with Streamlit:

```bash
# Auto-detect column names (recommended for most users)
streamlit run main.py

# Configure for different research data formats
streamlit run main.py -- --primary-id-column participant_id --session-column visit
streamlit run main.py -- --primary-id-column SubjectID --session-column Session --composite-id-column participantID

# Configure data directory and file settings
streamlit run main.py -- --data-dir custom_data --demographics-file my_demographics.csv

# Set UI defaults and performance options
streamlit run main.py -- --max-display-rows 100 --cache-ttl-seconds 300 --default-age-min 20 --default-age-max 70

# View all available options and examples
python main.py --help
```

#### Available CLI Parameters

**Data Structure Configuration:**
- `--primary-id-column`: Primary subject identifier column (default: 'ursi')
  - Common names: `participant_id`, `subject_id`, `SubjectID`, `study_id`
- `--session-column`: Session identifier for longitudinal data (default: 'session_num')
  - Common names: `session`, `timepoint`, `visit`, `Visit`, `Session`
- `--composite-id-column`: Generated composite ID column name (default: 'customID')

**File and Directory Settings:**
- `--data-dir`: Directory containing CSV data files (default: 'data')
- `--demographics-file`: Demographics CSV filename (default: 'demographics.csv')
- `--participant-id-column`: Legacy parameter for backward compatibility

**UI and Performance:**
- `--max-display-rows`: Maximum rows to display in preview (default: 50)
- `--cache-ttl-seconds`: Cache time-to-live in seconds (default: 600)
- `--default-age-min`: Default minimum age for age filter (default: 18)
- `--default-age-max`: Default maximum age for age filter (default: 80)

### Using the Interface

The application displays your detected data structure at startup (cross-sectional vs longitudinal) and shows any dataset preparation actions taken.

1. **Define Cohort Filters**:
   - Set age range filters (if age column exists)
   - Select sex categories (if sex column exists)
   - Select studies and sessions (for multi-study data)
   - Add phenotypic filters on numeric columns from any table
   - **Live participant count** updates as you adjust filters

2. **Select Data for Export**:
   - Choose which tables to include in your merged dataset
   - Select specific columns from each table
   - Tables are added to the bottom of the selection list
   - **Smart merging** uses appropriate strategy for your data format

3. **Generate & Download**:
   - **Enwiden by session** (longitudinal data only): Pivot data to wide format with session-specific columns
   - Click "Generate Merged Data" to run the query
   - Preview the first 50 rows of results
   - Download the complete dataset as a CSV file
   - Filename automatically reflects your filter settings and data format

## Configuration

### Runtime Configuration

The application can be configured at runtime using command line parameters (see [Command Line Configuration](#command-line-configuration) above).

### Code Configuration

Additional settings can be modified in the `Config` class in `main.py`:

- `SEX_MAPPING`: Mapping of sex labels to numeric codes
- `DEFAULT_AGE_RANGE`: Age slider range (default: 0-120)
- `DEFAULT_FILTER_RANGE`: Default range for phenotypic filters (default: 0-100)
- `SEX_OPTIONS`: Available sex options for UI selection
- `SESSION_OPTIONS`: Available session options for filtering

## Architecture

- **Frontend**: Streamlit web interface with reactive components
- **Backend**: DuckDB for in-memory SQL query processing
- **Data Processing**: Pandas for CSV handling and data manipulation
- **Caching**: Streamlit caching for performance optimization

### Key Functions

- `FlexibleMergeStrategy`: Auto-detects cross-sectional vs longitudinal data formats
- `MergeKeys`: Encapsulates merge column information and dataset structure  
- `get_table_info()`: Scans, analyzes, and prepares CSV files (cached for 10 minutes)
- `generate_base_query_logic()`: Creates adaptive SQL JOIN and WHERE clauses
- `render_*_filters()`: Modular UI components for different filter types
- `validate_csv_structure()`: Ensures data integrity with flexible column requirements
- `detect_rs1_format()`: Automatically detects multi-study format

## Performance

- **Optimized Loading**: Uses chunked reading for large files
- **Smart Caching**: Metadata cached separately from full data
- **Efficient Queries**: Parameterized SQL with DuckDB optimization
- **Memory Management**: Minimal memory footprint for large datasets

## Error Handling

The application includes comprehensive error handling for:
- Missing or malformed CSV files
- Invalid data types and formats
- Permission and file access issues
- SQL query errors and edge cases

## Development

### Code Structure

- `main.py`: Main application with UI and business logic
- `generate_synthetic_data.py`: Synthetic data generation for testing
- `requirements.txt`: Python package dependencies
- `pyproject.toml`: Project configuration and dependencies
- `CLAUDE.md`: Development guidelines for AI assistants

### Testing with Synthetic Data

```bash
# Generate synthetic test data
python generate_synthetic_data.py

# Run application with synthetic data
streamlit run main.py -- --data-dir synthetic_data
```

### Development Commands

```bash
# Install dependencies
uv sync

# Run the Streamlit application
streamlit run main.py

# Start Jupyter Lab for data analysis
jupyter lab
```

## License

This project is for research and educational use.
