# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based data browser application for laboratory research data. The app allows researchers to filter, query, merge, and export CSV datasets using an interactive web interface backed by DuckDB for efficient SQL queries.

## Key Architecture

- **Main Application**: `main.py` - Streamlit web interface for data exploration
- **Data Storage**: `data/` directory contains CSV files with research data
- **Flexible Merge Strategy**: Auto-detects cross-sectional (uses `ursi` or primary ID) vs longitudinal (creates `customID` from `ursi` + `session_num`)
- **Demographics Base**: `demographics.csv` serves as the primary table for LEFT JOINs
- **Query Engine**: DuckDB provides in-memory SQL processing for fast data operations

## Development Commands

```bash
# Install dependencies
uv sync

# Run the Streamlit application (auto-detects column names)
streamlit run main.py

# Common column name variations:
streamlit run main.py -- --primary-id-column subject_id --session-column timepoint
streamlit run main.py -- --primary-id-column participant_id --session-column visit
streamlit run main.py -- --primary-id-column SubjectID --session-column Session
streamlit run main.py -- --primary-id-column ursi --session-column session_num --composite-id-column participantID

# Different data directory
streamlit run main.py -- --data-dir /path/to/data --primary-id-column study_id

# Start Jupyter Lab for data analysis
jupyter lab
```

## Data Structure

The application automatically detects dataset structure and adapts merge strategy:

### Cross-sectional Data
- Uses primary ID column (default: `ursi`) for merging
- No session information required
- Simple, direct merging across tables

### Longitudinal Data  
- Detects presence of both primary ID (`ursi`) and session (`session_num`) columns
- Automatically creates `customID` = `ursi_session_num` for precise session-level merging
- Enables session-specific filtering and analysis

## Configuration Constants

- `DATA_DIR`: Directory containing CSV files (default: 'data')
- `DEMOGRAPHICS_FILE`: Primary demographics file (default: 'demographics.csv')
- `PRIMARY_ID_COLUMN`: Primary subject identifier (default: 'ursi', configurable via CLI)
- `SESSION_COLUMN`: Session identifier for longitudinal data (default: 'session_num', configurable via CLI)
- `COMPOSITE_ID_COLUMN`: Composite ID column name (default: 'customID', configurable via CLI)
- `PARTICIPANT_ID_COLUMN`: Auto-detected merge column (legacy support)
- `SEX_MAPPING`: Maps string sex values to numeric codes (Female=1.0, Male=2.0)

## CLI Configuration

The application supports extensive CLI configuration for different dataset formats:

```bash
# View all available options
python main.py --help

# Common research data naming conventions
streamlit run main.py -- --primary-id-column participant_id --session-column visit
streamlit run main.py -- --primary-id-column SubjectID --session-column Session --composite-id-column participantID
streamlit run main.py -- --primary-id-column study_id --session-column timepoint --data-dir /path/to/data
```

## Core Functions

- `FlexibleMergeStrategy`: Auto-detects and handles cross-sectional vs longitudinal data
- `MergeKeys`: Encapsulates merge column information and dataset structure
- `get_db_connection()`: Cached DuckDB connection
- `get_table_info()`: Scans data directory, detects structure, and returns metadata (cached 10 minutes)
- `generate_base_query_logic()`: Creates FROM/JOIN/WHERE clauses with flexible merge keys
- `generate_data_query()`: Builds full SELECT query for data export
- `generate_count_query()`: Builds COUNT query for participant matching
- `enwiden_longitudinal_data()`: Pivots longitudinal data from long to wide format for session-specific analysis

The application uses session state to manage table selection order and dynamic behavioral filters, with real-time participant count updates as filters are applied. It automatically displays the detected merge strategy to users for transparency. For longitudinal data, users can choose to export in either long format (one row per session) or wide format (one row per participant with session-specific columns like `age_BAS1`, `age_BAS2`).