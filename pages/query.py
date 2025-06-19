import dash
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import base64 # For decoding file contents
import io # For converting bytes to file-like object for pandas
import json # For JSON parsing
import logging
import duckdb
import pandas as pd
from datetime import datetime

# Assuming utils.py is in the same directory or accessible in PYTHONPATH
from utils import (
    Config,
    MergeKeys,
    validate_csv_file,
    save_uploaded_files_to_data_dir,
    get_table_info,
    detect_rs1_format,
    detect_rockland_format,
    is_numeric_column,
    generate_base_query_logic,
    generate_count_query,
    generate_data_query,
    enwiden_longitudinal_data
)

dash.register_page(__name__, path='/', title='Query Data')

# Initialize config instance
config = Config() # Loads from or creates config.toml

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Upload CSV Files"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=True # Allow multiple files to be uploaded
            ),
            html.Div(id='upload-status-container'), # Container for collapsible upload messages
            dcc.Store(id='upload-trigger-store'), # To trigger updates after successful uploads
# All persistent stores are now defined in the main app layout for cross-page access
            dcc.Store(id='rs1-checkbox-ids-store'), # To store IDs of dynamically generated RS1 checkboxes
            dash_table.DataTable(id='data-preview-table', style_table={'display': 'none'})
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Merge Strategy"),
            html.Div(id='merge-strategy-info'),
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Live Participant Count"),
            html.Div(id='live-participant-count'), # Placeholder for participant count
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Define Cohort Filters"),
            dbc.Card(dbc.CardBody([
                html.H4("Demographic Filters", className="card-title"),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Label("Age Range:"),
                        dcc.RangeSlider(id='age-slider', disabled=True, allowCross=False, step=1, tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(id='age-slider-info') # To show min/max or if disabled
                    ]), md=6),
                    dbc.Col(html.Div([
                        html.Label("Sex:"),
                        dcc.Dropdown(id='sex-dropdown', multi=True, disabled=True, placeholder="Select sex...")
                    ]), md=6),
                ]),
                html.Div(id='dynamic-demo-filters-placeholder', style={'marginTop': '20px'}), # For RS1, Rockland, Sessions
            ]), style={'marginTop': '20px'}),

            dbc.Card(dbc.CardBody([
                html.H4("Phenotypic Filters", className="card-title"),
                dbc.Button("Add Phenotypic Filter", id='add-phenotypic-filter-button', n_clicks=0, className="mb-3"),
                html.Div(id='phenotypic-filter-list'),
            ]), style={'marginTop': '20px'}),

        ], md=6), # Left column for filters
        dbc.Col([
            html.H3("Select Data for Export"),
            html.Div([
                html.H4("Select Tables:"),
                dcc.Dropdown(
                    id='table-multiselect',
                    multi=True,
                    placeholder="Select tables for export..."
                ),
                html.Div(id='column-selection-area'),
                html.Div([
                    dbc.Checkbox(
                        id='enwiden-data-checkbox',
                        label='Enwiden longitudinal data (pivot sessions to columns)',
                        value=False
                    )
                ], id='enwiden-checkbox-wrapper', style={'display': 'none', 'marginTop': '10px'})
            ], id='table-column-selector-container')
        ], md=6) # Right column for selections
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Query Results"),
            html.Div([
                dbc.Button(
                    "Generate Merged Data",
                    id='generate-data-button',
                    n_clicks=0,
                    color="primary",
                    className="mb-3"
                ),
                html.Div(id='data-preview-area'),
                dcc.Download(id='download-dataframe-csv')
            ], id='results-container')
        ], width=12)
    ])
], fluid=True)


# Callback to update Age Slider properties
@callback(
    [Output('age-slider', 'min'),
     Output('age-slider', 'max'),
     Output('age-slider', 'value'),
     Output('age-slider', 'marks'),
     Output('age-slider', 'disabled'),
     Output('age-slider-info', 'children')],
    [Input('demographics-columns-store', 'data'),
     Input('column-ranges-store', 'data')],
    [State('age-slider-state-store', 'data')]
)
def update_age_slider(demo_cols, col_ranges, stored_age_value):
    if not demo_cols or 'age' not in demo_cols or not col_ranges:
        return 0, 100, [0, 100], {}, True, "Age filter disabled: 'age' column not found in demographics or ranges not available."

    # Use 'demo' as the alias for demographics table, consistent with get_table_alias() in utils.py
    age_col_key = "demo.age" # Construct the key for column_ranges

    if age_col_key in col_ranges:
        min_age, max_age = col_ranges[age_col_key]
        min_age = int(min_age)
        max_age = int(max_age)

        default_min, default_max = config.DEFAULT_AGE_SELECTION
        
        # Use stored value if available and valid, otherwise use default
        if stored_age_value is not None and len(stored_age_value) == 2:
            stored_min, stored_max = stored_age_value
            if min_age <= stored_min <= max_age and min_age <= stored_max <= max_age:
                value = stored_age_value
            else:
                value = [max(min_age, default_min), min(max_age, default_max)]
        else:
            value = [max(min_age, default_min), min(max_age, default_max)]

        marks = {i: str(i) for i in range(min_age, max_age + 1, 10)}
        if min_age not in marks: marks[min_age] = str(min_age)
        if max_age not in marks: marks[max_age] = str(max_age)

        return min_age, max_age, value, marks, False, f"Age range: {min_age}-{max_age}"
    else:
        # Fallback if 'age' column is in demo_cols but no range found (should ideally not happen if get_table_info is robust)
        return 0, 100, [config.DEFAULT_AGE_SELECTION[0], config.DEFAULT_AGE_SELECTION[1]], {}, True, "Age filter disabled: Range for 'age' column not found."

# Callback to update Sex Dropdown properties
@callback(
    [Output('sex-dropdown', 'options'),
     Output('sex-dropdown', 'value'),
     Output('sex-dropdown', 'disabled')],
    [Input('demographics-columns-store', 'data')],
    [State('sex-dropdown-state-store', 'data')]
)
def update_sex_dropdown(demo_cols, stored_sex_value):
    if not demo_cols or 'sex' not in demo_cols:
        return [], None, True

    # Options from config.SEX_OPTIONS
    options = [{'label': s, 'value': s} for s in config.SEX_OPTIONS]
    
    # Use stored value if available, otherwise use default
    value = stored_sex_value if stored_sex_value is not None else config.DEFAULT_SEX_SELECTION
    return options, value, False

# Callback to populate dynamic demographic filters (RS1, Rockland, Sessions)
@callback(
    Output('dynamic-demo-filters-placeholder', 'children'),
    [Input('demographics-columns-store', 'data'),
     Input('session-values-store', 'data'),
     Input('merge-keys-store', 'data')],
    [State('rockland-substudy-store', 'data'),
     State('session-selection-store', 'data')]
)
def update_dynamic_demographic_filters(demo_cols, session_values, merge_keys_dict, 
                                     stored_rockland_values, stored_session_values):
    if not demo_cols:
        return html.P("Demographic information not yet available to populate dynamic filters.")

    children = []

    # RS1 Study Filters
    if detect_rs1_format(demo_cols, config): # utils.detect_rs1_format needs config
        children.append(html.H5("RS1 Study Selection", style={'marginTop': '15px'}))
        rs1_checkboxes = []
        for study_col, study_label in config.RS1_STUDY_LABELS.items():
            rs1_checkboxes.append(
                dbc.Checkbox(
                    id={'type': 'rs1-study-checkbox', 'index': study_col},
                    label=study_label,
                    value=study_col in config.DEFAULT_STUDY_SELECTION # Default checked based on config
                )
            )
        children.append(dbc.Form(rs1_checkboxes))

    # Rockland Substudy Filters
    if detect_rockland_format(demo_cols): # utils.detect_rockland_format
        children.append(html.H5("Substudy Selection", style={'marginTop': '15px'}))
        # Use stored values if available, otherwise use default
        rockland_value = stored_rockland_values if stored_rockland_values else config.DEFAULT_ROCKLAND_STUDIES
        children.append(
            dcc.Dropdown(
                id='rockland-substudy-dropdown',
                options=[{'label': s, 'value': s} for s in config.ROCKLAND_BASE_STUDIES],
                value=rockland_value,
                multi=True,
                placeholder="Select Rockland substudies..."
            )
        )

    # Session Filters
    if merge_keys_dict:
        mk = MergeKeys.from_dict(merge_keys_dict)
        if mk.is_longitudinal and mk.session_id and session_values:
            children.append(html.H5(f"{mk.session_id} Selection", style={'marginTop': '15px'}))
            # Use stored values if available, otherwise default to all available sessions
            session_value = stored_session_values if stored_session_values else session_values
            children.append(
                dcc.Dropdown(
                    id='session-dropdown',
                    options=[{'label': s, 'value': s} for s in session_values],
                    value=session_value,
                    multi=True,
                    placeholder=f"Select {mk.session_id} values..."
                )
            )

    if not children:
        return html.P("No dataset-specific demographic filters applicable.", style={'fontStyle': 'italic'})

    return children


# Callbacks to update stores when dynamic dropdowns change
@callback(
    Output('rockland-substudy-store', 'data'),
    Input('rockland-substudy-dropdown', 'value'),
    prevent_initial_call=True
)
def update_rockland_substudy_store(rockland_values):
    return rockland_values if rockland_values else []

@callback(
    Output('session-selection-store', 'data'),
    Input('session-dropdown', 'value'),
    prevent_initial_call=True
)
def update_session_selection_store(session_values):
    return session_values if session_values else []


# Callbacks for Phenotypic Filters

# Update the list of phenotypic filters displayed
@callback(
    Output('phenotypic-filter-list', 'children'),
    Input('phenotypic-filters-state-store', 'data'),
    State('available-tables-store', 'data'),
    State('behavioral-columns-store', 'data'),
    State('column-dtypes-store', 'data'),
    State('column-ranges-store', 'data'),
    State('demographics-columns-store', 'data'),
    State('merge-keys-store', 'data'),
)
def update_phenotypic_filter_list(
    filters_state, available_tables, behavioral_columns_by_table,
    column_dtypes, column_ranges, demo_columns, merge_keys_dict
):
    # This function dynamically generates the UI for each phenotypic filter
    # based on the current state of 'phenotypic-filters-state-store'.
    
    # Handle empty/missing data
    if not filters_state: 
        filters_state = []
    if not available_tables: 
        available_tables = []
    if not behavioral_columns_by_table: 
        behavioral_columns_by_table = {}
    if not column_dtypes: 
        column_dtypes = {}
    if not column_ranges: 
        column_ranges = {}
    if not demo_columns: 
        demo_columns = []

    merge_keys = MergeKeys.from_dict(merge_keys_dict) if merge_keys_dict else MergeKeys(primary_id="unknown")
    filter_elements = []
    demographics_table_name = config.get_demographics_table_name()

    # All tables available for filtering
    all_filterable_tables = [{'label': demographics_table_name, 'value': demographics_table_name}] + \
                            [{'label': table, 'value': table} for table in available_tables]

    for filter_config in filters_state:
        filter_id = filter_config['id']
        selected_table = filter_config.get('table')
        selected_column = filter_config.get('column')
        
        # Get available columns for the selected table
        all_columns_options = []
        if selected_table:
            table_actual_cols = []
            if selected_table == demographics_table_name:
                table_actual_cols = demo_columns
            elif selected_table in behavioral_columns_by_table:
                table_actual_cols = behavioral_columns_by_table[selected_table]

            id_cols_to_exclude = {merge_keys.primary_id, merge_keys.session_id, merge_keys.composite_id}
            
            for col in table_actual_cols:
                if col not in id_cols_to_exclude:
                    all_columns_options.append({'label': col, 'value': col})

        # Determine column type and create appropriate filter component
        filter_component = html.Div("Select table and column first", style={'color': 'gray', 'fontStyle': 'italic'})
        
        if selected_table and selected_column:
            table_alias = 'demo' if selected_table == demographics_table_name else selected_table
            dtype_key = f"{table_alias}.{selected_column}"
            column_dtype = column_dtypes.get(dtype_key)
            
            # logging.debug(f"Filter {filter_id} - table: {selected_table}, column: {selected_column}, dtype: {column_dtype}")
            
            if column_dtype and is_numeric_column(column_dtype):
                # Numeric column - use range slider
                range_key = f"{table_alias}.{selected_column}"
                if range_key in column_ranges:
                    min_val, max_val = column_ranges[range_key]
                    slider_min, slider_max = int(min_val), int(max_val)
                    current_min = filter_config.get('min_val', slider_min)
                    current_max = filter_config.get('max_val', slider_max)
                    slider_value = [current_min, current_max]
                    
                    filter_component = html.Div([
                        html.P(f"Range: {slider_min} - {slider_max}", style={'fontSize': '0.8em', 'margin': '0'}),
                        dcc.RangeSlider(
                            id={'type': 'pheno-range-slider', 'index': filter_id},
                            min=slider_min, max=slider_max, value=slider_value,
                            tooltip={"placement": "bottom", "always_visible": True},
                            allowCross=False, step=1,
                            marks={slider_min: str(slider_min), slider_max: str(slider_max)}
                        )
                    ])
                else:
                    filter_component = html.Div("No range data available for this numeric column", style={'color': 'orange'})
            else: # Categorical or other non-numeric column type
                unique_values, error_msg = get_unique_column_values(
                    data_dir=config.DATA_DIR, # Assuming global config instance
                    table_name=selected_table,
                    column_name=selected_column,
                    demo_table_name=config.get_demographics_table_name(),
                    demographics_file_name=config.DEMOGRAPHICS_FILE
                )
                if error_msg:
                    filter_component = html.Div(f"Error fetching values: {error_msg}", style={'color': 'red'})
                elif not unique_values:
                    filter_component = html.Div("No unique values found or column is empty.", style={'color': 'orange'})
                else:
                    filter_component = dcc.Dropdown(
                        id={'type': 'pheno-categorical-dropdown', 'index': filter_id},
                        options=[{'label': str(val), 'value': val} for val in unique_values], # Ensure label is string
                        value=filter_config.get('selected_values', []),
                        multi=True,
                        placeholder="Select value(s)..."
                    )

        filter_row = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Table:", style={'fontWeight': 'bold', 'fontSize': '0.9em'}),
                        dcc.Dropdown(
                            id={'type': 'pheno-table-dropdown', 'index': filter_id},
                            options=all_filterable_tables,
                            value=selected_table,
                            placeholder="Select Table"
                        )
                    ], width=3),
                    dbc.Col([
                        html.Label("Column:", style={'fontWeight': 'bold', 'fontSize': '0.9em'}),
                        dcc.Dropdown(
                            id={'type': 'pheno-column-dropdown', 'index': filter_id},
                            options=all_columns_options,
                            value=selected_column,
                            placeholder="Select Column",
                            disabled=not selected_table
                        )
                    ], width=3),
                    dbc.Col([
                        html.Label("Filter:", style={'fontWeight': 'bold', 'fontSize': '0.9em'}),
                        filter_component
                    ], width=5),
                    dbc.Col([
                        dbc.Button("Remove", 
                            id={'type': 'remove-pheno-filter-button', 'index': filter_id}, 
                            color="danger", size="sm", className="mt-4")
                    ], width=1)
                ])
            ])
        ], className="mb-3")
        
        filter_elements.append(filter_row)

    return filter_elements

# Add/Remove phenotypic filters based on button clicks
@callback(
    Output('phenotypic-filters-state-store', 'data'),
    Input('add-phenotypic-filter-button', 'n_clicks'),
    Input({'type': 'remove-pheno-filter-button', 'index': dash.ALL}, 'n_clicks'),
    State('phenotypic-filters-state-store', 'data')
)
def manage_phenotypic_filters_state(add_clicks, remove_clicks, current_filters):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Ensure current_filters is a list
    if current_filters is None:
        current_filters = []

    # logging.debug(f"Filter management - {button_id}, current count: {len(current_filters)}")

    if button_id == 'add-phenotypic-filter-button':
        # Generate a unique ID by finding the max existing ID and adding 1, ensuring it's an integer.
        existing_ids = [f['id'] for f in current_filters] if current_filters else []
        max_id = max(existing_ids) if existing_ids else 0
        new_filter_id = max_id + 1
        
        new_filter = {
            'id': new_filter_id,
            'table': None,
            'column': None,
            'filter_type': None,  # 'numeric' or 'categorical'
            # For numeric filters
            'min_val': None,
            'max_val': None,
            # For categorical filters
            'selected_values': None  # List of selected categorical values
        }
        # Create a new list instead of modifying the existing one
        updated_filters = current_filters.copy()
        updated_filters.append(new_filter)
        # logging.debug(f"Added filter ID {new_filter_id}, total: {len(updated_filters)}")
        return updated_filters
    else: # Must be a remove button
        try:
            # Attempt to parse the button_id string as a JSON object
            # This is expected for pattern-matching IDs like {'type': 'remove-pheno-filter-button', 'index': 1}
            clicked_button_dict = json.loads(button_id)
            if isinstance(clicked_button_dict, dict) and clicked_button_dict.get('type') == 'remove-pheno-filter-button':
                filter_id_to_remove = clicked_button_dict['index']
                # Ensure comparison is safe (e.g. if filter_id_to_remove could be string and f['id'] integer)
                # Assuming IDs are consistently integers.
                updated_filters = [f for f in current_filters if f['id'] != filter_id_to_remove]
                return updated_filters
        except json.JSONDecodeError:
            # This path might be taken if button_id is not a JSON string (e.g. simple string ID from a non-pattern-matching component)
            # Or if the n_clicks for the remove buttons are not correctly processed (they should be)
            logging.warning(f"Could not parse button_id as JSON or identify remove action: {button_id}")
        except Exception as e:
            logging.exception(f"Error processing remove filter button click for ID {button_id}: {e}") # Log full traceback
            return current_filters # Return original filters on error

    return current_filters # Default return if no action matched

# Update column dropdown options when table selection changes in a phenotypic filter
@callback(
    Output({'type': 'pheno-column-dropdown', 'index': dash.MATCH}, 'options'),
    Output({'type': 'pheno-column-dropdown', 'index': dash.MATCH}, 'value'), # Reset column when table changes
    Output({'type': 'pheno-column-dropdown', 'index': dash.MATCH}, 'disabled'),
    Input({'type': 'pheno-table-dropdown', 'index': dash.MATCH}, 'value'),
    State('demographics-columns-store', 'data'),
    State('behavioral-columns-store', 'data'),
    State('column-dtypes-store', 'data'),
    State('merge-keys-store', 'data')
)
def update_pheno_column_options(selected_table, demo_cols, behavioral_cols, col_dtypes, merge_keys_dict):
    if not selected_table or not col_dtypes:
        return [], None, True

    options = []
    demographics_table_name = config.get_demographics_table_name()
    table_actual_cols = []

    if selected_table == demographics_table_name:
        table_actual_cols = demo_cols if demo_cols else []
    elif selected_table in behavioral_cols:
        table_actual_cols = behavioral_cols[selected_table]

    merge_keys = MergeKeys.from_dict(merge_keys_dict) if merge_keys_dict else MergeKeys(primary_id="unknown")
    id_cols_to_exclude = {merge_keys.primary_id, merge_keys.session_id, merge_keys.composite_id}

    for col in table_actual_cols:
        if col in id_cols_to_exclude:
            continue
        # Allow all non-ID columns to be selectable for filtering
        # The type of filter (numeric/categorical) will be determined by update_phenotypic_filter_list
        options.append({'label': col, 'value': col})

    return options, None, not bool(options)


# Update range slider properties when column selection changes in a phenotypic filter
@callback(
    [Output({'type': 'pheno-range-slider', 'index': dash.MATCH}, 'min'),
     Output({'type': 'pheno-range-slider', 'index': dash.MATCH}, 'max'),
     Output({'type': 'pheno-range-slider', 'index': dash.MATCH}, 'value'),
     Output({'type': 'pheno-range-slider', 'index': dash.MATCH}, 'disabled')],
    Input({'type': 'pheno-column-dropdown', 'index': dash.MATCH}, 'value'),
    State({'type': 'pheno-table-dropdown', 'index': dash.MATCH}, 'value'),
    State('column-ranges-store', 'data'),
    State('phenotypic-filters-state-store', 'data'), # To get stored min/max for this filter
    State({'type': 'pheno-range-slider', 'index': dash.MATCH}, 'id'), # To get the filter_id
    prevent_initial_call=True
)
def update_pheno_range_slider(selected_column, selected_table, col_ranges, filters_state, slider_id_dict):
    if not selected_column or not selected_table or not col_ranges:
        return 0, 100, [0, 100], True

    filter_id = slider_id_dict['index']
    current_filter_config = None
    if filters_state:
        current_filter_config = next((f for f in filters_state if f['id'] == filter_id), None)

    demographics_table_name = config.get_demographics_table_name()
    table_alias = 'demo' if selected_table == demographics_table_name else selected_table
    range_key = f"{table_alias}.{selected_column}"

    if range_key in col_ranges:
        min_val, max_val = col_ranges[range_key]
        min_r, max_r = int(min_val), int(max_val)

        # Use stored values if available and valid, else full range
        stored_min = current_filter_config.get('min_val') if current_filter_config else None
        stored_max = current_filter_config.get('max_val') if current_filter_config else None

        current_val = [
            stored_min if stored_min is not None and stored_min >= min_r and stored_min <= max_r else min_r,
            stored_max if stored_max is not None and stored_max <= max_r and stored_max >= min_r else max_r
        ]
        # Ensure min <= max
        if current_val[0] > current_val[1]: current_val[0] = current_val[1]

        # logging.debug(f"Range slider update - filter_id: {filter_id}, range: [{min_r}, {max_r}], current_val: {current_val}")
        return min_r, max_r, current_val, False
    return 0, 100, [0, 100], True


# REMOVED: Problematic initialization callback that was causing conflicts


# Simplified callback for updating filter state - handles one type at a time to prevent conflicts
@callback(
    Output('phenotypic-filters-state-store', 'data', allow_duplicate=True),
    Input({'type': 'pheno-table-dropdown', 'index': dash.ALL}, 'value'),
    State('phenotypic-filters-state-store', 'data'),
    State({'type': 'pheno-table-dropdown', 'index': dash.ALL}, 'id'),
    prevent_initial_call=True
)
def update_filter_table_selection(table_values, current_filters_state, dropdown_ids):
    """Update filter state when table is selected"""
    if not current_filters_state or not table_values or not dropdown_ids:
        return dash.no_update
    
    # logging.debug(f"Table update - values: {table_values}")
    
    updated_filters = []
    for i, (table_val, dropdown_id) in enumerate(zip(table_values, dropdown_ids)):
        filter_id = dropdown_id['index']
        
        # Find matching filter in current state
        matching_filter = next((f for f in current_filters_state if f['id'] == filter_id), None)
        if matching_filter:
            updated_filter = matching_filter.copy()
            if updated_filter.get('table') != table_val:
                # Table changed - reset other fields
                updated_filter['table'] = table_val
                updated_filter['column'] = None
                updated_filter['filter_type'] = None
                updated_filter['min_val'] = None
                updated_filter['max_val'] = None
                updated_filter['selected_values'] = None
            updated_filters.append(updated_filter)
    
    # Add any filters that don't have corresponding dropdowns (shouldn't happen but for safety)
    existing_filter_ids = {f['id'] for f in updated_filters}
    for f in current_filters_state:
        if f['id'] not in existing_filter_ids:
            updated_filters.append(f)
    
    return updated_filters

# REMOVE START: update_filter_column_selection
# @callback(
#     Output('phenotypic-filters-state-store', 'data', allow_duplicate=True),
#     Input({'type': 'pheno-column-dropdown', 'index': dash.ALL}, 'value'),
#     State('phenotypic-filters-state-store', 'data'),
#     State({'type': 'pheno-column-dropdown', 'index': dash.ALL}, 'id'),
#     prevent_initial_call=True
# )
# def update_filter_column_selection(column_values, current_filters_state, dropdown_ids):
#     """Update filter state when column is selected"""
#     if not current_filters_state or not column_values or not dropdown_ids:
#         return dash.no_update
    
#     # logging.debug(f"Column update - values: {column_values}")
    
#     updated_filters = []
#     for i, (column_val, dropdown_id) in enumerate(zip(column_values, dropdown_ids)):
#         filter_id = dropdown_id['index']
        
#         # Find matching filter in current state
#         matching_filter = next((f for f in current_filters_state if f['id'] == filter_id), None)
#         if matching_filter:
#             updated_filter = matching_filter.copy()
#             if updated_filter.get('column') != column_val:
#                 # Column changed - reset filter values but keep table
#                 updated_filter['column'] = column_val
#                 updated_filter['filter_type'] = None  # Will be determined by display callback
#                 updated_filter['min_val'] = None
#                 updated_filter['max_val'] = None
#                 updated_filter['selected_values'] = None
#             updated_filters.append(updated_filter)
    
#     # Add any filters that don't have corresponding dropdowns
#     existing_filter_ids = {f['id'] for f in updated_filters}
#     for f in current_filters_state:
#         if f['id'] not in existing_filter_ids:
#             updated_filters.append(f)
    
#     return updated_filters
# REMOVE END: update_filter_column_selection

# REMOVE START: update_filter_range_values
# @callback(
#     Output('phenotypic-filters-state-store', 'data', allow_duplicate=True),
#     Input({'type': 'pheno-range-slider', 'index': dash.ALL}, 'value'),
#     State('phenotypic-filters-state-store', 'data'),
#     State({'type': 'pheno-range-slider', 'index': dash.ALL}, 'id'),
#     prevent_initial_call=True
# )
# def update_filter_range_values(range_values, current_filters_state, slider_ids):
#     """Update filter state when range slider values change"""
#     if not current_filters_state or not range_values or not slider_ids:
#         return dash.no_update

#     # logging.debug(f"Range update - values: {range_values}")

#     updated_filters = []
#     for current_filter in current_filters_state:
#         updated_filter = current_filter.copy()

#         # Find corresponding range slider
#         for range_val, slider_id in zip(range_values, slider_ids):
#             if slider_id['index'] == current_filter['id']:
#                 if range_val and len(range_val) == 2:
#                     updated_filter['min_val'] = range_val[0]
#                     updated_filter['max_val'] = range_val[1]
#                     updated_filter['filter_type'] = 'numeric'
#                 break

#         updated_filters.append(updated_filter)

#     return updated_filters
# REMOVE END: update_filter_range_values

# Unified callback for updating phenotypic filter properties
@callback(
    Output('phenotypic-filters-state-store', 'data', allow_duplicate=True),
    Input({'type': 'pheno-table-dropdown', 'index': dash.ALL}, 'value'),
    Input({'type': 'pheno-column-dropdown', 'index': dash.ALL}, 'value'),
    Input({'type': 'pheno-range-slider', 'index': dash.ALL}, 'value'),
    Input({'type': 'pheno-categorical-dropdown', 'index': dash.ALL}, 'value'), # New Input
    State('phenotypic-filters-state-store', 'data'),
    prevent_initial_call=True
)
def update_phenotypic_filter_properties(
    table_values, column_values, range_values, categorical_values, # New argument
    current_filters_state
):
    ctx = dash.callback_context
    if not ctx.triggered or not current_filters_state:
        return dash.no_update

    triggered_component = ctx.triggered[0]
    prop_id = triggered_component['prop_id']
    triggered_value = triggered_component['value']
    
    # The ID of the component that triggered the callback is ctx.triggered_id
    # For pattern-matching callbacks, ctx.triggered_id is expected to be a dictionary.
    # e.g., {'type': 'pheno-table-dropdown', 'index': 1}
    
    triggered_id_dict = ctx.triggered_id
    if not isinstance(triggered_id_dict, dict):
        # This should not happen for pattern-matching ALL callbacks if the trigger is one of the pattern-matched inputs.
        # It might happen if a non-pattern-matched Input was somehow also part of ctx.triggered,
        # or if an Input is not actually a pattern-matching ID.
        # For safety, we check.
        logging.warning(f"Unified callback - ctx.triggered_id is not a dict: {triggered_id_dict}. prop_id was {prop_id}")
        # Attempt to parse from prop_id as a fallback, though this is less robust.
        try:
            # prop_id is like "{'type':'pheno-table-dropdown','index':0}.value"
            id_str = prop_id.split('.')[0]
            if isinstance(id_str, str) and '{' in id_str: # Check if it looks like a JSON string
                 triggered_id_dict = json.loads(id_str) # Parse it
            else: # If not a JSON string, cannot proceed with this fallback.
                logging.error(f"Unified callback - prop_id does not seem to contain a JSON object: {id_str}")
                return dash.no_update
        except json.JSONDecodeError: # If JSON parsing fails
            logging.exception(f"Unified callback - JSONDecodeError parsing prop_id: {prop_id}")
            return dash.no_update
        
    filter_id_to_update = triggered_id_dict.get('index')
    component_type = triggered_id_dict.get('type')

    # Validate that filter_id_to_update and component_type are present
    if filter_id_to_update is None or not component_type:
        logging.error(f"Unified callback - Missing 'index' or 'type' in triggered_id_dict: {triggered_id_dict}")
        return dash.no_update

    # logging.debug(f"Unified callback - Filter ID: {filter_id_to_update}, Type: {component_type}, Value: {triggered_value}")

    # Create a new list to store updated filters
    updated_filters_list = []
    found_filter = False

    for f_config in current_filters_state:
        current_filter_copy = f_config.copy()
        if current_filter_copy['id'] == filter_id_to_update:
            found_filter = True
            if component_type == 'pheno-table-dropdown':
                if current_filter_copy.get('table') != triggered_value:
                    current_filter_copy['table'] = triggered_value
                    current_filter_copy['column'] = None
                    current_filter_copy['filter_type'] = None
                    current_filter_copy['min_val'] = None
                    current_filter_copy['max_val'] = None
                    current_filter_copy['selected_values'] = None
                    # logging.debug(f"Unified - Updated table for filter {filter_id_to_update}")
            elif component_type == 'pheno-column-dropdown':
                if current_filter_copy.get('column') != triggered_value:
                    current_filter_copy['column'] = triggered_value
                    current_filter_copy['filter_type'] = None # Will be set by display logic or slider
                    current_filter_copy['min_val'] = None
                    current_filter_copy['max_val'] = None
                    current_filter_copy['selected_values'] = None
                    # logging.debug(f"Unified - Updated column for filter {filter_id_to_update}")
            elif component_type == 'pheno-range-slider':
                if triggered_value and len(triggered_value) == 2:
                    new_min, new_max = triggered_value
                    if current_filter_copy.get('min_val') != new_min or current_filter_copy.get('max_val') != new_max:
                        current_filter_copy['min_val'] = new_min
                        current_filter_copy['max_val'] = new_max
                        current_filter_copy['filter_type'] = 'numeric'
                        # logging.debug(f"Unified - Updated range for filter {filter_id_to_update}")
            elif component_type == 'pheno-categorical-dropdown':
                selected_cats = triggered_value # This is the list of selected values
                # Check if selection actually changed
                if current_filter_copy.get('selected_values') != selected_cats:
                    current_filter_copy['selected_values'] = selected_cats
                    current_filter_copy['filter_type'] = 'categorical'
                    current_filter_copy['min_val'] = None # Clear numeric filter props
                    current_filter_copy['max_val'] = None # Clear numeric filter props
                    # logging.debug(f"Unified - Updated categorical selection for filter {filter_id_to_update}")

        updated_filters_list.append(current_filter_copy)

    if not found_filter:
        # This case should ideally not happen if states are consistent
        logging.warning(f"Unified callback - Filter ID {filter_id_to_update} not found in state.")
        return dash.no_update
        
    return updated_filters_list


# Live Participant Count Callback
@callback(
    Output('live-participant-count', 'children'),
    [Input('age-slider', 'value'),
     Input('sex-dropdown', 'value'),
     Input({'type': 'rs1-study-checkbox', 'index': dash.ALL}, 'value'), # For RS1 studies
     Input('rockland-substudy-store', 'data'), # For Rockland substudies
     Input('session-selection-store', 'data'), # For session filtering
     Input('phenotypic-filters-state-store', 'data'),
     # Data stores needed for query generation
     Input('merge-keys-store', 'data'),
     Input('available-tables-store', 'data')] # To determine tables to join for phenotypic filters
)
def update_live_participant_count(
    age_range, selected_sex,
    rs1_study_values, # RS1 studies from dynamic checkboxes
    rockland_substudy_values, # Rockland substudies from store
    session_values, # Session values from store
    phenotypic_filters_state,
    merge_keys_dict, available_tables
):
    ctx = dash.callback_context
    if not ctx.triggered and not merge_keys_dict : # Don't run on initial load if no data yet
        return dbc.Alert("Upload data and select filters to see participant count.", color="info")

    if not merge_keys_dict:
        return dbc.Alert("Merge strategy not determined. Cannot calculate count.", color="warning")

    current_config = config # Use global config instance
    merge_keys = MergeKeys.from_dict(merge_keys_dict)

    demographic_filters = {}
    if age_range:
        demographic_filters['age_range'] = age_range
    if selected_sex:
        demographic_filters['sex'] = selected_sex

    # Collect RS1 study selections
    # Assuming rs1_study_values corresponds to the 'value' (boolean) of dbc.Checkbox
    # and ctx.inputs_list[2] gives us the list of component states that includes their IDs.
    # This part is a bit complex due to dynamic component IDs.
    # A simpler way if IDs are predictable:
    selected_rs1_studies = []
    if ctx.inputs_list and len(ctx.inputs_list) > 2:
        rs1_input_states = ctx.inputs_list[2] # List of {'id': {'index': 'is_DS', 'type': 'rs1-study-checkbox'}, 'value': True/False}
        for i, state in enumerate(rs1_input_states):
            # The actual value comes from rs1_study_values[i]
            # The id comes from state['id']['index']
            if rs1_study_values[i]: # if checkbox is checked
                 selected_rs1_studies.append(state['id']['index'])
    if selected_rs1_studies:
        demographic_filters['studies'] = selected_rs1_studies

    # Handle Rockland substudy filtering
    if rockland_substudy_values:
        demographic_filters['substudies'] = rockland_substudy_values
        
    # Handle session filtering
    if session_values:
        demographic_filters['sessions'] = session_values

    # Determine active phenotypic filters based on filter type
    active_phenotypic_filters = []
    for f in phenotypic_filters_state:
        if f.get('table') and f.get('column'):
            if f.get('filter_type') == 'numeric' and f.get('min_val') is not None and f.get('max_val') is not None:
                active_phenotypic_filters.append(f)
            elif f.get('filter_type') == 'categorical' and f.get('selected_values'):
                active_phenotypic_filters.append(f)
    
    # logging.debug(f"Live count - Total phenotypic filters: {len(phenotypic_filters_state) if phenotypic_filters_state else 0}")
    # logging.debug(f"Live count - Active phenotypic filters: {len(active_phenotypic_filters)}")
    # for f_idx, f_val in enumerate(active_phenotypic_filters):
    #     logging.debug(f"DEBUG: Active filter {f_idx}: {f_val}")

    # Determine tables to join: must include demographics, and any table mentioned in phenotypic filters.
    # Also, if session filters are active, and it's longitudinal, ensure at least one behavioral table is joined
    # if session_id is primarily on those.
    tables_for_query = {current_config.get_demographics_table_name()}
    for p_filter in active_phenotypic_filters:
        tables_for_query.add(p_filter['table'])

    # Add a behavioral table if session filter is active and data is longitudinal,
    # and only demo table is currently selected for query.
    # This logic is simplified here. A more robust way would be to check if session_id column exists in tables.
    if merge_keys.is_longitudinal and demographic_filters.get('sessions') and len(tables_for_query) == 1 and available_tables:
        # Add first available behavioral table to enable session filtering if it's not already there.
        # This assumes session_id might not be in demo table or its filtering is linked to behavioral tables.
        if available_tables[0] not in tables_for_query: # Add first one if not demo
             tables_for_query.add(available_tables[0])


    try:
        base_query, params = generate_base_query_logic(
            current_config, merge_keys, demographic_filters, active_phenotypic_filters, list(tables_for_query)
        )
        count_query, count_params = generate_count_query(base_query, params, merge_keys)

        if count_query:
            # Establish a new connection for each callback execution for safety in threaded Dash environment
            # For very high frequency updates, a shared connection with appropriate locking might be considered.
            with duckdb.connect(database=':memory:', read_only=False) as con:
                count_result = con.execute(count_query, count_params).fetchone()

            if count_result and count_result[0] is not None:
                return dbc.Alert(f"Matching Rows: {count_result[0]}", color="success")
            else:
                return dbc.Alert("Could not retrieve participant count.", color="warning")
        else:
            return dbc.Alert("No query generated for count.", color="info")

    except Exception as e:
        logging.error(f"Error during live count query: {e}")
        logging.error(f"Query attempted: {count_query if 'count_query' in locals() else 'N/A'}")
        logging.error(f"Params: {count_params if 'count_params' in locals() else 'N/A'}")
        return dbc.Alert(f"Error calculating count: {str(e)}", color="danger")


@callback(
    [Output('upload-status-container', 'children'),
     Output('upload-trigger-store', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')],
    prevent_initial_call=True
)
def create_collapsible_upload_messages(messages, num_files=0):
    """Create a collapsible component for upload messages"""
    if not messages:
        return html.Div()
    
    # Count different message types
    validation_msgs = [msg for msg in messages if hasattr(msg, 'children') and 'is valid' in str(msg.children)]
    save_msgs = [msg for msg in messages if hasattr(msg, 'children') and ('Saved' in str(msg.children) or 'Error' in str(msg.children))]
    error_msgs = [msg for msg in messages if hasattr(msg, 'style') and msg.style.get('color') == 'red']
    
    # Summary line
    summary_text = f"Processed {num_files} files"
    if error_msgs:
        summary_text += f" ({len(error_msgs)} errors)"
    
    summary_color = "danger" if error_msgs else "success"
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-upload me-2"),
                summary_text,
                dbc.Button(
                    [html.I(className="fas fa-chevron-down")],
                    id="upload-messages-toggle",
                    color="link",
                    size="sm",
                    className="float-end p-0",
                    style={"border": "none"}
                ),
                dbc.Button(
                    [html.I(className="fas fa-times")],
                    id="upload-messages-dismiss",
                    color="link",
                    size="sm", 
                    className="float-end p-0 me-2",
                    style={"border": "none", "color": "red"}
                )
            ], className="mb-0")
        ], className=f"bg-{summary_color} text-white"),
        dbc.Collapse([
            dbc.CardBody([
                html.Div(messages, style={"max-height": "300px", "overflow-y": "auto"})
            ])
        ], id="upload-messages-collapse", is_open=False)
    ], className="mb-3")

def handle_file_uploads(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        return html.Div("No files uploaded."), dash.no_update

    messages = []
    all_files_valid = True
    saved_file_names = []
    file_byte_contents = []

    if list_of_names:
        for i, (c, n, d) in enumerate(zip(list_of_contents, list_of_names, list_of_dates)):
            try:
                content_type, content_string = c.split(',')
                decoded = base64.b64decode(content_string)

                # Validate each file
                validation_errors, df = validate_csv_file(decoded, n) # Using the new signature

                if validation_errors:
                    all_files_valid = False
                    for error in validation_errors:
                        messages.append(html.Div(f"Error with {n}: {error}", style={'color': 'red'}))
                else:
                    messages.append(html.Div(f"File {n} is valid.", style={'color': 'green'}))
                    file_byte_contents.append(decoded)
                    saved_file_names.append(n) # Keep track of names for saving

            except Exception as e:
                all_files_valid = False
                messages.append(html.Div(f"Error processing file {n}: {str(e)}", style={'color': 'red'}))
                continue # Skip to next file if this one errors out during processing

    if all_files_valid and file_byte_contents:
        # Save valid files
        # utils.save_uploaded_files_to_data_dir expects lists of contents and names
        success_msgs, error_msgs = save_uploaded_files_to_data_dir(file_byte_contents, saved_file_names, config.DATA_DIR)
        for msg in success_msgs:
            messages.append(html.Div(msg, style={'color': 'green'}))
        for err_msg in error_msgs:
            messages.append(html.Div(err_msg, style={'color': 'red'}))

        num_files = len(list_of_names) if list_of_names else 0
        collapsible_messages = create_collapsible_upload_messages(messages, num_files)
        
        if not error_msgs: # Only trigger if all saves were successful
             # Trigger downstream updates by changing the store's data
            return collapsible_messages, {'timestamp': datetime.now().isoformat()}
        else:
            return collapsible_messages, dash.no_update

    elif not file_byte_contents: # No valid files to save
        messages.append(html.Div("No valid files were processed to save.", style={'color': 'orange'}))
        num_files = len(list_of_names) if list_of_names else 0
        collapsible_messages = create_collapsible_upload_messages(messages, num_files)
        return collapsible_messages, dash.no_update
    else: # Some files were invalid
        num_files = len(list_of_names) if list_of_names else 0
        collapsible_messages = create_collapsible_upload_messages(messages, num_files)
        return collapsible_messages, dash.no_update


# Callbacks for collapsible upload messages
@callback(
    Output('upload-messages-collapse', 'is_open'),
    [Input('upload-messages-toggle', 'n_clicks')],
    [State('upload-messages-collapse', 'is_open')],
    prevent_initial_call=True
)
def toggle_upload_messages(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@callback(
    Output('upload-status-container', 'children', allow_duplicate=True),
    [Input('upload-messages-dismiss', 'n_clicks')],
    prevent_initial_call=True
)
def dismiss_upload_messages(n_clicks):
    if n_clicks:
        return html.Div()
    return dash.no_update


@callback(
    [Output('available-tables-store', 'data'),
     Output('demographics-columns-store', 'data'),
     Output('behavioral-columns-store', 'data'),
     Output('column-dtypes-store', 'data'),
     Output('column-ranges-store', 'data'),
     Output('merge-keys-store', 'data'),
     Output('session-values-store', 'data'),
     Output('all-messages-store', 'data'), # To display errors from get_table_info
     Output('merge-strategy-info', 'children')],
    [Input('upload-trigger-store', 'data'), # Triggered by successful file upload
     Input('upload-data', 'contents')] # Also trigger on initial page load if files are "already there" (less common for upload)
)
def load_initial_data_info(trigger_data, _): # trigger_data from upload-trigger-store, _ for upload-data contents (initial)
    # We need to use the global config instance that was loaded/created when query.py was imported.
    # Or, if config can change dynamically (e.g. via UI), it needs to be managed in a dcc.Store

    # Re-fetch config if it could have changed (e.g., if settings were editable in another part of the app)
    # For now, assume config loaded at app start is sufficient, or re-initialize.
    current_config = config # Use global config instance

    (behavioral_tables, demographics_cols, behavioral_cols_by_table,
     col_dtypes, col_ranges, merge_keys_dict,
     actions_taken, session_vals, is_empty, messages) = get_table_info(current_config)

    info_messages = []
    if messages: # 'messages' is 'all_messages' from get_table_info, which is List[str]
        for msg_text in messages:
            color = "black" # Default color
            msg_lower = msg_text.lower()
            if "error" in msg_lower:
                color = "red"
            elif "warning" in msg_lower or "warn" in msg_lower: # Catch 'warning' or 'warn'
                color = "orange"
            elif "info" in msg_lower or "note" in msg_lower: # Catch 'info' or 'note'
                color = "blue"
            elif "✅" in msg_text or "success" in msg_lower: # Catch success indicators
                color = "green"

            info_messages.append(html.P(msg_text, style={'color': color}))

    if actions_taken:
        info_messages.append(html.H5("Dataset Preparation Actions:", style={'marginTop': '10px'}))
        for action in actions_taken:
            info_messages.append(html.P(action))

    merge_strategy_display = [html.H5("Merge Strategy:", style={'marginTop': '10px'})]
    if merge_keys_dict:
        mk = MergeKeys.from_dict(merge_keys_dict)
        if mk.is_longitudinal:
            merge_strategy_display.append(html.P(f"Detected: Longitudinal data."))
            merge_strategy_display.append(html.P(f"Primary ID: {mk.primary_id}"))
            merge_strategy_display.append(html.P(f"Session ID: {mk.session_id}"))
            merge_strategy_display.append(html.P(f"Composite ID (for merge): {mk.composite_id}"))
        else:
            merge_strategy_display.append(html.P(f"Detected: Cross-sectional data."))
            merge_strategy_display.append(html.P(f"Primary ID (for merge): {mk.primary_id}"))
    else:
        merge_strategy_display.append(html.P("Merge strategy not determined yet. Upload data or check configuration."))

    # Combine messages from get_table_info with other status messages
    # Place dataset preparation actions and merge strategy info after general messages.
    status_display_content = info_messages + merge_strategy_display
    status_display = html.Div(status_display_content)


    return (behavioral_tables, demographics_cols, behavioral_cols_by_table,
            col_dtypes, col_ranges, merge_keys_dict, session_vals,
            messages, # Store raw messages from get_table_info for potential detailed display
            status_display) # This now goes to 'merge-strategy-info' Div


# Callbacks for Table and Column Selection
@callback(
    Output('table-multiselect', 'options'),
    Input('available-tables-store', 'data')
)
def update_table_multiselect_options(available_tables_data):
    if not available_tables_data:
        return []
    # available_tables_data is a list of table names (strings)
    return [{'label': table, 'value': table} for table in available_tables_data]

@callback(
    Output('table-multiselect', 'value', allow_duplicate=True),
    Input('available-tables-store', 'data'),
    State('table-multiselect-state-store', 'data'),
    prevent_initial_call=True
)
def restore_table_multiselect_value(available_tables_data, stored_value):
    if stored_value is not None:
        return stored_value
    return []

@callback(
    Output('enwiden-data-checkbox', 'value', allow_duplicate=True),
    Input('merge-keys-store', 'data'),
    State('enwiden-data-checkbox-state-store', 'data'),
    prevent_initial_call=True
)
def restore_enwiden_checkbox_value(merge_keys_dict, stored_value):
    if stored_value is not None:
        return stored_value
    return False

@callback(
    Output('column-selection-area', 'children'),
    Input('table-multiselect', 'value'), # List of selected table names
    State('demographics-columns-store', 'data'),
    State('behavioral-columns-store', 'data'),
    State('merge-keys-store', 'data'),
    State('selected-columns-per-table-store', 'data') # To pre-populate selections
)
def update_column_selection_area(selected_tables, demo_cols, behavioral_cols, merge_keys_dict, stored_selections):
    if not selected_tables:
        return dbc.Alert("Select tables above to choose columns for export.", color="info")

    if not demo_cols: demo_cols = []
    if not behavioral_cols: behavioral_cols = {}
    if not stored_selections: stored_selections = {}

    merge_keys = MergeKeys.from_dict(merge_keys_dict) if merge_keys_dict else MergeKeys(primary_id="unknown")
    id_cols_to_exclude = {merge_keys.primary_id, merge_keys.session_id, merge_keys.composite_id}
    demographics_table_name = config.get_demographics_table_name()

    cards = []
    for table_name in selected_tables:
        options = []
        actual_cols_for_table = []
        is_demographics_table = (table_name == demographics_table_name)

        if is_demographics_table:
            actual_cols_for_table = demo_cols
        elif table_name in behavioral_cols:
            actual_cols_for_table = behavioral_cols[table_name]

        for col in actual_cols_for_table:
            if col not in id_cols_to_exclude: # Exclude ID columns from selection
                options.append({'label': col, 'value': col})

        # Get previously selected columns for this table, if any
        current_selection_for_table = stored_selections.get(table_name, [])

        card_body_content = [
            dcc.Dropdown(
                id={'type': 'column-select-dropdown', 'table': table_name},
                options=options,
                value=current_selection_for_table, # Pre-populate with stored selections
                multi=True,
                placeholder=f"Select columns from {table_name}..."
            )
        ]
        # If it's the demographics table, add a note that ID columns are auto-included
        if is_demographics_table:
            card_body_content.insert(0, html.P(f"All demographics columns (including {merge_keys.get_merge_column()}) will be included by default. You can select additional ones if needed, or deselect to only include IDs/merge keys.", className="small text-muted"))


        cards.append(dbc.Card([
            dbc.CardHeader(f"Columns for: {table_name}"),
            dbc.CardBody(card_body_content)
        ], className="mb-3"))

    return cards

@callback(
    Output('selected-columns-per-table-store', 'data'),
    Input({'type': 'column-select-dropdown', 'table': dash.ALL}, 'value'), # Values from all column dropdowns
    State({'type': 'column-select-dropdown', 'table': dash.ALL}, 'id'), # IDs of all column dropdowns
    State('selected-columns-per-table-store', 'data') # Current stored data
)
def update_selected_columns_store(all_column_values, all_column_ids, current_stored_data):
    ctx = dash.callback_context
    
    # Make a copy to modify, or initialize if None
    updated_selections = current_stored_data.copy() if current_stored_data else {}

    # Only update if callback was actually triggered by user interaction
    # This prevents overwriting stored data on initial page load
    if ctx.triggered and all_column_ids and all_column_values:
        for i, component_id_dict in enumerate(all_column_ids):
            table_name = component_id_dict['table']
            selected_cols_for_table = all_column_values[i]

            if selected_cols_for_table is not None: # An empty selection is an empty list, None means no interaction yet
                updated_selections[table_name] = selected_cols_for_table
            elif table_name in updated_selections and selected_cols_for_table is None:
                # This case might occur if a table is deselected from table-multiselect,
                # its column dropdown might fire a final None value.
                # However, the update_column_selection_area callback should remove the dropdown.
                # If a user manually clears a dropdown, it becomes an empty list.
                pass # No change if value is None and table already existed or didn't.
    else:
        # Return no_update to preserve stored data when callback isn't triggered by user interaction
        return no_update

    # This ensures that if a table is deselected from 'table-multiselect',
    # its column selections are removed from the store.
    # We get the list of currently *rendered* tables from the IDs.
    # Any table in 'updated_selections' NOT in this list should be removed.
    current_rendered_tables = {comp_id['table'] for comp_id in all_column_ids}
    keys_to_remove = [table_key for table_key in updated_selections if table_key not in current_rendered_tables]
    for key in keys_to_remove:
        del updated_selections[key]

    return updated_selections

# Callback to control "Enwiden Data" checkbox visibility
@callback(
    Output('enwiden-checkbox-wrapper', 'style'), # Target the wrapper div
    Input('merge-keys-store', 'data')
)
def update_enwiden_checkbox_visibility(merge_keys_dict):
    if merge_keys_dict:
        mk = MergeKeys.from_dict(merge_keys_dict)
        if mk.is_longitudinal:
            return {'display': 'block', 'marginTop': '10px'} # Show
    return {'display': 'none'} # Hide

# Callback for Data Generation
@callback(
    [Output('data-preview-area', 'children'),
     Output('merged-dataframe-store', 'data')], # Store for profiling page
    Input('generate-data-button', 'n_clicks'),
    [State('age-slider', 'value'),
     State('sex-dropdown', 'value'),
     State({'type': 'rs1-study-checkbox', 'index': dash.ALL}, 'value'),
     State('rockland-substudy-store', 'data'),
     State('session-selection-store', 'data'),
     State('phenotypic-filters-state-store', 'data'),
     State('selected-columns-per-table-store', 'data'),
     State('enwiden-data-checkbox', 'value'), # Boolean value (True when checked, False when unchecked)
     State('merge-keys-store', 'data'),
     State('available-tables-store', 'data'), # Needed for tables_to_join logic
     State('table-multiselect', 'value')] # Explicitly selected tables for export
)
def handle_generate_data(
    n_clicks,
    age_range, selected_sex, rs1_study_values,
    rockland_substudy_values, session_filter_values,
    phenotypic_filters_state, selected_columns_per_table,
    enwiden_checkbox_value, merge_keys_dict, available_tables, tables_selected_for_export
):
    if n_clicks == 0 or not merge_keys_dict:
        return dbc.Alert("Click 'Generate Merged Data' after selecting filters and columns.", color="info"), None

    current_config = config # Use global config instance
    merge_keys = MergeKeys.from_dict(merge_keys_dict)

    # --- Collect Demographic Filters ---
    demographic_filters = {}
    if age_range: demographic_filters['age_range'] = age_range
    if selected_sex: demographic_filters['sex'] = selected_sex

    selected_rs1_studies = []
    # Accessing rs1_study_values directly as it's a list of booleans from the checkboxes
    # Need to map these back to the study column names using the order from config.RS1_STUDY_LABELS
    # This assumes the order of checkboxes matches RS1_STUDY_LABELS.items()
    # A more robust way would be to get the component IDs if they were static or use ctx.inputs_list carefully.
    # For dynamically generated checkboxes via a callback, their state needs to be carefully managed.
    # Let's assume the `update_dynamic_demographic_filters` callback ensures IDs are {'type': 'rs1-study-checkbox', 'index': study_col}
    # and rs1_study_values is a list of their `value` properties (True/False)

    # Simplified: If rs1_study_values is available and True, get its corresponding ID.
    # This requires that the Input for rs1_study_values provides enough context or
    # that we fetch rs1_study_ids from another source (e.g. a hidden store updated by dynamic filter callback)
    # For now, we'll rely on the direct values if they are simple lists of selected items.
    # If rs1_study_values is a list of booleans, we need to associate them with the study names.
    # This part is tricky with dash.ALL for dynamically generated checkboxes if not handled carefully.
    # A common pattern is to have the callback that generates these also store their IDs/relevant info.
    # Given the current structure, we assume rs1_study_values are the *values* of the checked items,
    # and we need their corresponding *IDs* (study column names).
    # This part of the logic might need refinement based on how `{'type': 'rs1-study-checkbox', 'index': dash.ALL}` actually passes data.
    # It typically passes a list of the `property` specified (here, 'value').
    # We need the 'index' part of the ID for those that are True.
    # This requires iterating through `dash.callback_context.inputs_list` or `triggered` if it's an Input.
    # For now, let's assume `rs1_study_values` contains the `study_col` if checked.
    # This would be true if the `value` property of dbc.Checkbox was set to `study_col` itself.
    # Rechecking the dynamic filter callback: value is set to `study_col in config.DEFAULT_STUDY_SELECTION`
    # This means rs1_study_values is a list of booleans.
    # We need the corresponding 'index' from the ID for those that are True.
     # This is now handled by taking rs1_checkbox_ids_store as State.

    # Correctly accessing RS1 study selections:
    # For now, get the study names from config based on checkbox values
    if rs1_study_values:
        study_cols = list(config.RS1_STUDY_LABELS.keys())
        selected_rs1_studies = [study_cols[i] for i, checked in enumerate(rs1_study_values) if i < len(study_cols) and checked]
        if selected_rs1_studies:
            demographic_filters['studies'] = selected_rs1_studies

    # Handle Rockland substudy filtering
    if rockland_substudy_values:
        demographic_filters['substudies'] = rockland_substudy_values
        
    # Handle session filtering
    if session_filter_values:
        demographic_filters['sessions'] = session_filter_values

    # --- Phenotypic Filters ---
    active_phenotypic_filters = []
    for f in phenotypic_filters_state:
        if f.get('table') and f.get('column'):
            if f.get('filter_type') == 'numeric' and f.get('min_val') is not None and f.get('max_val') is not None:
                active_phenotypic_filters.append(f)
            elif f.get('filter_type') == 'categorical' and f.get('selected_values'):
                active_phenotypic_filters.append(f)

    # --- Determine Tables to Join ---
    # Start with tables explicitly selected for export
    tables_for_query = set(tables_selected_for_export if tables_selected_for_export else [])
    tables_for_query.add(current_config.get_demographics_table_name()) # Always include demographics
    for p_filter in active_phenotypic_filters: # Add tables from active phenotypic filters
        tables_for_query.add(p_filter['table'])

    # Note: Session-based table selection logic would go here when session filters are available
    # if merge_keys.is_longitudinal and demographic_filters.get('sessions') and \
    #    len(tables_for_query.intersection(set(available_tables if available_tables else []))) == 0 and \
    #    current_config.get_demographics_table_name() in tables_for_query and \
    #    len(tables_for_query) == 1 and available_tables:
    #     tables_for_query.add(available_tables[0]) # Add a behavioral table if needed for session join

    # --- Selected Columns for Query ---
    # If no columns are selected for a table in tables_selected_for_export, select all its non-ID columns.
    # For tables *only* in phenotypic filters (not for export), we don't need to select their columns explicitly for SELECT clause,
    # as they are just for filtering via JOIN. Demo table columns are handled by demo.*

    query_selected_columns = selected_columns_per_table.copy() if selected_columns_per_table else {}
    # For tables selected for export but with no specific columns chosen, this implies "all columns" for that table.
    # The generate_data_query handles this: if a table is in selected_tables (i.e. tables_for_query here)
    # and has entries in selected_columns, those are used. If demo.* is default, other tables need explicit columns.
    # For this implementation, we assume selected_columns_per_table_store correctly reflects user choices.
    # If a table is in tables_selected_for_export, it should be in query_selected_columns.
    # If user wants all columns from table X, they should use "select all" in its column dropdown (not yet implemented).
    # For now, only explicitly selected columns via UI are passed. generate_data_query adds demo.*

    try:
        base_query, params = generate_base_query_logic(
            current_config, merge_keys, demographic_filters, active_phenotypic_filters, list(tables_for_query)
        )
        data_query, data_params = generate_data_query(
            base_query, params, list(tables_for_query), query_selected_columns
        )

        if not data_query:
            return dbc.Alert("Could not generate data query.", color="warning"), None

        with duckdb.connect(database=':memory:', read_only=False) as con:
            result_df = con.execute(data_query, data_params).fetchdf()

        original_row_count = len(result_df)

        if enwiden_checkbox_value and merge_keys.is_longitudinal:
            result_df = enwiden_longitudinal_data(result_df, merge_keys)
            enwiden_info = f" (enwidened from {original_row_count} rows to {len(result_df)} rows)"
        else:
            enwiden_info = ""

        if result_df.empty:
            return dbc.Alert("No data found for the selected criteria.", color="info"), None

        # Prepare for DataTable
        dt_columns = [{"name": i, "id": i} for i in result_df.columns]
        # For performance, only show head in preview
        dt_data = result_df.head(current_config.MAX_DISPLAY_ROWS).to_dict('records')

        preview_table = dash_table.DataTable(
            data=dt_data,
            columns=dt_columns,
            page_size=10,
            style_table={'overflowX': 'auto'},
            filter_action="native",
            sort_action="native",
        )

        return (
            html.Div([
                dbc.Alert(f"Query successful. Displaying first {min(len(result_df), current_config.MAX_DISPLAY_ROWS)} of {len(result_df)} total rows{enwiden_info}.", color="success"),
                preview_table,
                html.Hr(),
                dbc.Button("Download CSV", id="download-csv-button", color="success", className="mt-2")
            ]),
            result_df.to_dict('records') # Store full data for profiling page (consider size limits)
        )

    except Exception as e:
        logging.error(f"Error during data generation: {e}")
        logging.error(f"Query attempted: {data_query if 'data_query' in locals() else 'N/A'}")
        return dbc.Alert(f"Error generating data: {str(e)}", color="danger"), None


# Callback for CSV Download Button
@callback(
    Output('download-dataframe-csv', 'data'),
    Input('download-csv-button', 'n_clicks'),
    [State('merged-dataframe-store', 'data'),
     State('age-slider', 'value')],
    prevent_initial_call=True
)
def download_csv_data(n_clicks, stored_data, age_range):
    if n_clicks is None or not stored_data:
        return dash.no_update
    
    # Convert stored data back to DataFrame
    df = pd.DataFrame(stored_data)
    
    # Create a filename
    filename_parts = ["merged_data"]
    if age_range: 
        filename_parts.append(f"age{age_range[0]}-{age_range[1]}")
    filename = "_".join(filename_parts) + ".csv"
    
    return dcc.send_data_frame(df.to_csv, filename, index=False)


# Unified callback to save all filter states for persistence across page navigation
@callback(
    [Output('age-slider-state-store', 'data'),
     Output('sex-dropdown-state-store', 'data'),
     Output('table-multiselect-state-store', 'data'),
     Output('enwiden-data-checkbox-state-store', 'data')],
    [Input('age-slider', 'value'),
     Input('sex-dropdown', 'value'),
     Input('table-multiselect', 'value'),
     Input('enwiden-data-checkbox', 'value')]
)
def save_all_filter_states(age_value, sex_value, table_value, enwiden_value):
    return age_value, sex_value, table_value, enwiden_value
