# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based data browser application for laboratory research data. The app allows researchers to filter, query, merge, and export CSV datasets using an interactive web interface backed by DuckDB for efficient SQL queries.

## Key Architecture

- **Main Application**: `app.py` - Streamlit web interface for data exploration
- **Data Storage**: `data/` directory contains CSV files with research data
- **Primary Key**: All CSV files use `customID` as the participant identifier for joining
- **Demographics Base**: `demographics.csv` serves as the primary table for LEFT JOINs
- **Query Engine**: DuckDB provides in-memory SQL processing for fast data operations

## Development Commands

```bash
# Install dependencies
uv sync

# Run the Streamlit application
streamlit run app.py

# Run the simple main script
python main.py

# Start Jupyter Lab for data analysis
jupyter lab
```

## Data Structure

The application expects CSV files in the `data/` directory with a common `customID` column for participant identification. The demographics table serves as the base for all joins, with other behavioral/assessment tables joined as needed.

## Configuration Constants

- `DATA_DIR`: Directory containing CSV files (currently 'data')
- `DEMOGRAPHICS_FILE`: Primary demographics file ('demographics.csv')
- `PARTICIPANT_ID_COLUMN`: Common identifier across tables ('customID')
- `SEX_MAPPING`: Maps string sex values to numeric codes (Female=1.0, Male=2.0)

## Core Functions

- `get_db_connection()`: Cached DuckDB connection
- `get_table_info()`: Scans data directory and returns table metadata (cached 10 minutes)
- `generate_base_query_logic()`: Creates FROM/JOIN/WHERE clauses for filtering
- `generate_data_query()`: Builds full SELECT query for data export
- `generate_count_query()`: Builds COUNT query for participant matching

The application uses session state to manage table selection order and dynamic behavioral filters, with real-time participant count updates as filters are applied.