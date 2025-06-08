# Lab Data Query and Merge Tool

A Streamlit-based web application for laboratory research data exploration, filtering, and export. This tool allows researchers to interactively query, merge, and download CSV datasets using an intuitive interface backed by DuckDB for efficient data processing.

## Who Needs This Application?

You have data from multiple outcome measures stored in multiple CSV files. You'd like to be able to filter out some participants, select outcome variables from several CSVs, and then merge those data into a wide-format CSV for further analysis. You are smart enough to avoid copy/paste operations in Excel.

This application enables efficient Filter, Query, and Merge operations across your data tables saved in standard CSV files with **no need to create a SQL database or stand up a SQL server process**. Just direct the application to your data folder, and then filter, query, and merge to pull your desired analytic dataset from the larger database - much as you would in LORIS or RedCap.

**Key advantages:**
- **No dependence on your IT department** - this lightweight application runs on your laptop
- **Easy data updates** - to update your database, just update your CSV(s), drop them in your application's data folder, and restart the application
- **No database administration** - no need to negotiate with your IT department or cajole a tech-savvy colleague to update your SQL database
- **Familiar workflow** - works like web-based research databases you already know, but with complete local control

## Features

- **Interactive Data Filtering**: Apply demographic and phenotypic filters to identify participant cohorts
- **Real-time Participant Count**: See matching participant counts update as you adjust filters
- **Table Merging**: Join multiple CSV datasets on a common participant identifier
- **Flexible Column Selection**: Choose specific columns from each table for export
- **Fast Performance**: Optimized data loading and query execution using DuckDB
- **Export Functionality**: Download filtered and merged datasets as CSV files
- **RS1 Study Support**: Built-in support for multi-study datasets with session filtering
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
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Create synthetic data (optional)**
   
   To test the interface without adding 'real' data you can run:
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

## Data Setup

1. Place your CSV files in the `data/` directory (or specify a custom directory)
2. Ensure all CSV files contain a common participant identifier column (Default=`customID`, but can be configured using the flag `--participant-id-column`; see 'Available CLI Parameters', below.
3. Include a `demographics.csv` file as the primary table for joins
4. Optionally include `age` and `sex` columns in demographics for demographic filtering

### Example Data Structure

```
data/
├── demographics.csv      # Primary table (required)
├── VO2max.csv           # Phenotypic data table
├── woodcock_johnson.csv # Cognitive data table
├── flanker.csv          # Behavioral data table
└── ...                  # Additional data tables
```

### Data Requirements

- **Required**: All CSV files must have a `customID` column for participant identification
- **Demographics table**: Must be named `demographics.csv` (or specified via CLI)
- **Optional columns in demographics**: `age`, `sex` for demographic filtering
- **RS1 format support**: Automatically detected if demographics contains study columns (`is_DS`, `is_ALG`, `is_CLG`, `is_NFB`)

## Usage

### Running the Application

```bash
# Start the Streamlit application
streamlit run main.py

# Start Jupyter Lab for data analysis
jupyter lab
```

### Command Line Configuration

The application supports runtime configuration via command line parameters. Use the double dash (`--`) separator when running with Streamlit:

```bash
# Configure data directory and file settings
streamlit run main.py -- --data-dir custom_data --demographics-file my_demo.csv

# Set UI defaults and performance options
streamlit run main.py -- --max-display-rows 100 --cache-ttl-seconds 300

# Configure participant ID column and age defaults
streamlit run main.py -- --participant-id-column ID --default-age-min 20 --default-age-max 70

# View all available options
streamlit run main.py -- --help
```

#### Available CLI Parameters

- `--data-dir`: Directory containing CSV data files (default: 'data')
- `--demographics-file`: Demographics CSV filename (default: 'demographics.csv')  
- `--participant-id-column`: Column name for participant ID across all tables (default: 'customID')
- `--max-display-rows`: Maximum rows to display in preview (default: 50)
- `--cache-ttl-seconds`: Cache time-to-live in seconds (default: 600)
- `--default-age-min`: Default minimum age for age filter (default: 18)
- `--default-age-max`: Default maximum age for age filter (default: 80)

### Using the Interface

1. **Define Cohort Filters**:
   - Set age range filters (if age column exists)
   - Select sex categories (if sex column exists)
   - Select studies and sessions (for RS1 format data)
   - Add phenotypic filters on numeric columns from any table

2. **Select Data for Export**:
   - Choose which tables to include in your merged dataset
   - Select specific columns from each table
   - Tables are added to the bottom of the selection list

3. **Generate & Download**:
   - Click "Generate Merged Data" to run the query
   - Preview the first 50 rows of results
   - Download the complete dataset as a CSV file

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

- `get_table_info()`: Scans and analyzes CSV files (cached for 10 minutes)
- `generate_base_query_logic()`: Creates SQL JOIN and WHERE clauses
- `render_*_filters()`: Modular UI components for different filter types
- `validate_csv_structure()`: Ensures data integrity and required columns
- `detect_rs1_format()`: Automatically detects RS1 study format

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