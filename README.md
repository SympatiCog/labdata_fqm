# Data Browser

A Streamlit-based web application for laboratory research data exploration, filtering, and export. This tool allows researchers to interactively query, merge, and download CSV datasets using an intuitive interface backed by DuckDB for efficient data processing.

## Features

- **Interactive Data Filtering**: Apply demographic and behavioral filters to identify participant cohorts
- **Real-time Participant Count**: See matching participant counts update as you adjust filters
- **Table Merging**: Join multiple CSV datasets on a common participant identifier
- **Flexible Column Selection**: Choose specific columns from each table for export
- **Fast Performance**: Optimized data loading and query execution using DuckDB
- **Export Functionality**: Download filtered and merged datasets as CSV files

## Prerequisites

- Python 3.11 or higher
- UV package manager (recommended) or pip

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Data Setup

1. Place your CSV files in the `data/` directory
2. Ensure all CSV files contain a common participant identifier column named `customID`
3. Include a `demographics.csv` file as the primary table for joins
4. Optionally include `age` and `sex` columns in demographics for demographic filtering

### Example Data Structure

```
data/
├── demographics.csv      # Primary table (required)
├── VO2max.csv           # Behavioral data table
├── woodcock_johnson.csv # Behavioral data table
└── ...                  # Additional data tables
```

## Usage

### Running the Application

```bash
# Start the Streamlit application
streamlit run app.py

# Or run the simple demo script
python main.py

# Start Jupyter Lab for data analysis
jupyter lab
```

### Command Line Configuration

The application supports runtime configuration via command line parameters. Use the double dash (`--`) separator when running with Streamlit:

```bash
# Configure data directory and file settings
streamlit run app.py -- --data-dir custom_data --demographics-file my_demo.csv

# Set UI defaults and performance options
streamlit run app.py -- --max-display-rows 100 --cache-ttl-seconds 300

# Configure participant ID column and age defaults
streamlit run app.py -- --participant-id-column ID --default-age-min 20 --default-age-max 70

# View all available options
streamlit run app.py -- --help
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
   - Add behavioral filters on numeric columns from any table

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

Additional settings can be modified in the `Config` class in `app.py`:

- `SEX_MAPPING`: Mapping of sex labels to numeric codes
- `DEFAULT_AGE_RANGE`: Age slider range (default: 0-120)
- `DEFAULT_FILTER_RANGE`: Default range for behavioral filters (default: 0-100)
- `SEX_OPTIONS`: Available sex options for UI selection

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

- `app.py`: Main application with UI and business logic
- `main.py`: Simple demo script
- `pyproject.toml`: Project dependencies and configuration
- `CLAUDE.md`: Development guidelines for AI assistants

### Testing

```bash
# Syntax validation
python -c "import app; print('Syntax check passed')"

# Run the application locally
streamlit run app.py
```

## License

This project is for research and educational use.