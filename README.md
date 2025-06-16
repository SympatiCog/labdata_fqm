# The Basic Scientist's Basic Data Tool (Plotly Dash Version)

A Plotly Dash-based web application for laboratory research data filtering, querying, merging, and comprehensive data profiling. This tool allows researchers to interactively query, merge, download queried datasets, and generate detailed profiling reports for CSV datasets using an intuitive multipage interface, backed by DuckDB for efficient data processing.

## Who Wants/Needs This Application?

* You have research data stored in multiple CSV files.
* You'd like to be able to:
    * Select a subset of variables from across your CSVs.
    * Merge those variables into a single wide-format CSV for further analysis.
    * Filter out participants based on various criteria.
* **You prefer a GUI over manual scripting for these tasks and want to avoid error-prone spreadsheet operations.**
* You want the power and efficiency of SQL without needing to set up or manage a traditional SQL database/server.

**Key advantages:**
- **Local Operation**: Runs on your local machine, no IT department dependency for core use.
- **Easy Data Updates**: Update your data by modifying your CSV files in the designated data folder.
- **No Database Administration**: Leverages DuckDB for on-the-fly SQL processing of CSVs.
- **Familiar Workflow**: Provides an interactive experience for data manipulation and analysis.

## Features

### 🔍 **Data Query & Merging**
- **Smart Data Structure Detection**: Automatically detects cross-sectional vs. longitudinal data formats.
- **Flexible Column Configuration**: Adapts to common column naming conventions for participant IDs and session identifiers.
- **Interactive Data Filtering**: Apply demographic (age, sex, study-specific) and phenotypic filters (numeric ranges from any table).
- **Real-time Participant Count**: See matching participant counts update as you adjust filters.
- **Intelligent Table Merging**: Merges data based on detected or configured merge keys.
- **Flexible Column Selection**: Choose specific columns from each table for export.
- **Data Pivoting (Enwiden)**: Transform longitudinal data from long to wide format (e.g., `age_BAS1`, `age_BAS2`).
- **Fast Performance**: Utilizes DuckDB for efficient query execution on CSVs.
- **Export Functionality**: Download filtered and merged datasets as CSV files.

### 📊 **Data Profiling & Analysis**
- **Comprehensive Data Profiling**: Generate detailed statistical analysis reports using `ydata-profiling`.
- **Interactive Visualizations**: Includes correlation matrices, distribution plots, missing values heatmaps, etc., within the report.
- **Multiple Report Types**: Select Full, Minimal, or Explorative profiling modes.
- **Performance Optimized**: Option to use sampling for large datasets.
- **Export Reports**: Download detailed HTML and JSON profiling reports.
- **Standalone Analysis**: Upload CSV files directly on the profiling page for analysis.
- **Data Quality Assessment**: Identify missing values, outliers, and data type issues through the report.

### 🏗️ **Application Structure**
- **Multipage Interface**: Organized navigation between Data Query and Data Profiling pages.
- **Session State Management**: Shares merged data from the Query page to the Profiling page.
- **Responsive Design**: Built with Dash Bootstrap Components for usability on various screen sizes.

### ⚙️ **Configuration & Management**
- **TOML-based Configuration**: Uses `config.toml` for persistent settings like data directory, default column names, and UI preferences.
- **Automatic Creation**: `config.toml` is created with default values on first run if not present.

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SympatiCog/labdata_fqm.git
    cd labdata_fqm
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Unix/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Ensure your CSV data files are in a directory** (default is `data/` in the project root).
    *   Create a `data` directory in the project root if it doesn't exist.
    *   Place your CSV files inside. One file should be your main demographics table (e.g., `demographics.csv`).
    *   The application will attempt to auto-detect participant IDs and session columns. You can customize these in `config.toml`.

2.  **Run the Dash application:**
    ```bash
    python app.py
    ```

3.  **Open your web browser** and navigate to `http://127.0.0.1:8050/`.

## Configuration (`config.toml`)

The application uses a `config.toml` file for configuration. If this file does not exist when the application starts, it will be created automatically with default settings.

**Key configuration options:**

*   `DATA_DIR`: Path to your CSV data files (e.g., "data", "my_research_data/csvs").
*   `DEMOGRAPHICS_FILE`: Filename of your primary demographics CSV (e.g., "demographics.csv").
*   `PRIMARY_ID_COLUMN`: Default name for the primary subject identifier column (e.g., "ursi", "subject_id").
*   `SESSION_COLUMN`: Default name for the session identifier column for longitudinal data (e.g., "session_num", "visit").
*   `COMPOSITE_ID_COLUMN`: Default name for the column that will store the combined ID+Session for merging longitudinal data (e.g., "customID").
*   `DEFAULT_AGE_SELECTION`: Default age range selected in the UI (e.g., `[18, 80]`).
*   `SEX_MAPPING`: Mapping for 'sex' column values to numerical representations if needed by your data.

You can edit `config.toml` directly to change these settings. The application reads this file on startup.

## Project Structure

*   `app.py`: Main Dash application entry point, defines the overall app layout and navbar.
*   `pages/`: Directory containing individual page modules for the Dash app.
    *   `query.py`: Logic and layout for the Data Query & Merge page.
    *   `profiling.py`: Logic and layout for the Data Profiling page.
*   `utils.py`: Utility functions, including data processing, query generation, and configuration management.
*   `assets/`: Directory for CSS or JavaScript files (if any).
*   `data/`: Default directory for user's CSV data files (can be changed in `config.toml`).
*   `config.toml`: Configuration file (auto-generated if not present).
*   `requirements.txt`: Python package dependencies.
*   `README.md`: This file.

## Development

To set up for development, follow the installation instructions above.
The application runs in debug mode by default when using `python app.py`, which enables hot-reloading.

## License

This project is for research and educational use.
