import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Query Data", href="/")),
            dbc.NavItem(dbc.NavLink("Profile Data", href="/profiling")),
        ],
        brand="Data Query and Profiling Tool",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-2",
    ),
    dash.page_container
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)
