import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Custom Color Palette
COLORS = {
    "cyan": "#00f5ff",
    "coral": "#ff6b6b",
    "gold": "#ffd700",
    "purple": "#b23ada",
    "green": "#00ff7f",
    "bg": "#0d1117",
    "paper": "#161b22",
    "grid": "#2b303b"
}

def apply_custom_theme(fig):
    fig.update_layout(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["paper"],
        font=dict(color="#e0e0e0", family="Poppins, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, color="#aaa"),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, color="#aaa"),
        title_font=dict(color="#ffffff", size=15, family="Rajdhani, sans-serif"),
    )
    return fig

def plot_3d_surface(df, countries, metric="new_cases"):
    """
    3D Surface Plot: X=Time, Y=Country, Z=Metric
    """
    df_filtered = df[df["location"].isin(countries)].copy()
    
    # Resample to weekly to keep surface plot performant
    df_filtered = df_filtered.set_index("date")
    # Pivot table: rows = date, cols = country, values = metric
    pivot_df = pd.pivot_table(df_filtered, values=metric, index="date", columns="location", aggfunc="sum")
    
    # Resample weekly
    pivot_df = pivot_df.resample('W').sum().fillna(0)
    
    # Define XYZ
    x = pivot_df.index
    y = pivot_df.columns
    z = pivot_df.values.T # Shape: (len(y), len(x))
    
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])

    fig.update_layout(
        title=f"3D Surface: {metric.replace('_', ' ').title()} by Country",
        scene=dict(
            xaxis_title='Date', yaxis_title='Country', zaxis_title=metric,
            bgcolor="#0d1117",
            xaxis=dict(gridcolor="#2b303b"),
            yaxis=dict(gridcolor="#2b303b"),
            zaxis=dict(gridcolor="#2b303b"),
        ),
        paper_bgcolor=COLORS["paper"]
    )
    fig.update_layout(font=dict(color="#e0e0e0"))
    return fig

def plot_animated_choropleth(df, metric="total_cases"):
    """
    World map showing total cases animated by month.
    """
    # Exclude continents from global map
    df_countries = df[df["continent"].notna()].copy()
    
    # Downsample to monthly for animation frames, taking the max value of the month
    df_countries["month"] = df_countries["date"].dt.to_period("M").astype(str)
    df_monthly = df_countries.groupby(["location", "iso_code", "month"], as_index=False)[metric].max()
    df_monthly = df_monthly.sort_values(["month", "location"])
    
    fig = px.choropleth(
        df_monthly,
        locations="iso_code",
        color=metric,
        hover_name="location",
        animation_frame="month",
        color_continuous_scale="Plasma",
        title=f"Global Spread ({metric.replace('_', ' ').title()})"
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False, showcoastlines=True,
            projection_type='equirectangular',
            bgcolor=COLORS["bg"],
            lakecolor=COLORS["bg"]
        ),
        paper_bgcolor=COLORS["paper"],
        font=dict(color="#e0e0e0")
    )
    return fig

def plot_dual_axis_line(df_country, metric="new_cases", smooth=True):
    """
    Dual axis: Daily metric (bars or line) + 7-day rolling average
    """
    country_name = df_country["location"].iloc[0]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Primary axis - Raw data
    fig.add_trace(
        go.Bar(x=df_country["date"], y=df_country[metric], name="Raw Daily", marker_color=COLORS["cyan"], opacity=0.4),
        secondary_y=False
    )
    
    # Secondary axis - Rolling average
    if smooth:
        rolling_avg = df_country[metric].rolling(window=7).mean()
        fig.add_trace(
            go.Scatter(x=df_country["date"], y=rolling_avg, name="7-Day Avg", line=dict(color=COLORS["coral"], width=3)),
            secondary_y=True
        )
        
    fig.update_layout(title=f"Dual-Axis Plot: {metric.replace('_', ' ').title()} - {country_name}")
    fig.update_yaxes(title_text="Raw Data", secondary_y=False)
    fig.update_yaxes(title_text="7-Day Avg", secondary_y=True)
    return apply_custom_theme(fig)

def plot_stacked_area(df_country):
    """
    Stacked Area Chart: Active Cases vs Deaths over time.
    (Note: Recoveries are not reliably provided in OWID, so we proxy active/deaths)
    """
    country_name = df_country["location"].iloc[0]
    
    fig = go.Figure()
    
    # Total Deaths
    fig.add_trace(go.Scatter(
        x=df_country["date"], y=df_country["total_deaths"],
        mode='lines', line=dict(width=0.5, color=COLORS["coral"]),
        fill='tozeroy', name='Total Deaths'
    ))
    
    # Active Cases
    fig.add_trace(go.Scatter(
        x=df_country["date"], y=df_country["active_cases"],
        mode='lines', line=dict(width=0.5, color=COLORS["cyan"]),
        fill='tonexty', name='Active Cases Proxy'
    ))
    
    fig.update_layout(title=f"Cumulative Profile: {country_name}")
    return apply_custom_theme(fig)

def plot_vaccination_race(df, top_n=10):
    """
    Horizontal animated bar chart for % vaccinated.
    """
    df_countries = df[df["continent"].notna()].copy()
    
    # Downsample to monthly taking max per month
    df_countries["month"] = df_countries["date"].dt.to_period("M").astype(str)
    
    # Calculate % vaccinated based on people_fully_vaccinated / population (safeguard divide by zero)
    df_countries["pct_vaccinated"] = np.where(
        df_countries["population"] > 0, 
        (df_countries["people_fully_vaccinated"] / df_countries["population"]) * 100,
        0
    )
    
    df_monthly = df_countries.groupby(["location", "month"], as_index=False)["pct_vaccinated"].max()
    df_monthly["pct_vaccinated"] = df_monthly["pct_vaccinated"].clip(upper=100) # Cap at 100%
    
    # Keep only Top N countries per month for animation performance
    def get_top_n(group):
        return group.nlargest(top_n, "pct_vaccinated")
        
    df_top = df_monthly.groupby("month").apply(get_top_n).reset_index(drop=True)
    df_top = df_top.sort_values(["month", "pct_vaccinated"], ascending=[True, True])
    
    fig = px.bar(
        df_top, 
        x="pct_vaccinated", 
        y="location", 
        color="location",
        animation_frame="month", 
        orientation='h', 
        range_x=[0, 100],
        title="Vaccination Progress Bar Race (%)"
    )
    
    fig.update_layout(showlegend=False)
    return apply_custom_theme(fig)

def plot_correlation_heatmap(df_country):
    """
    Correlation matrix: cases, deaths, vaccinations, stringency, gdp
    """
    country_name = df_country["location"].iloc[0]
    
    cols = ["new_cases", "new_deaths", "people_fully_vaccinated", "stringency_index", "gdp_per_capita"]
    corr_df = df_country[cols].corr().fillna(0)
    
    labels = ["Cases", "Deaths", "Vaccinations", "Stringency", "GDP"]
    
    fig = px.imshow(
        corr_df, 
        x=labels, 
        y=labels, 
        color_continuous_scale="Viridis",
        title=f"Correlation Heatmap: {country_name}"
    )
    
    return apply_custom_theme(fig)

def plot_forecast(forecast_df, country_name, metric, metrics_dict, actual_df):
    """
    Plots the forecast with confidence intervals.
    actual_df is the full historical data up to May 2023 for visual context.
    """
    fig = go.Figure()
    
    # Plot recent actual data (e.g., last 90 days of the series)
    recent_actual = actual_df.iloc[-90:]
    
    fig.add_trace(go.Scatter(
        x=recent_actual["date"], y=recent_actual[metric],
        mode='lines', name='Actual Data', line=dict(color=COLORS['cyan'])
    ))
    
    # Plot forecasted mean
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"],
        mode='lines', name='Forecast', line=dict(color=COLORS['gold'], dash='dash')
    ))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
        y=list(forecast_df["upper_bound"]) + list(forecast_df["lower_bound"])[::-1],
        fill='toself', fillcolor='rgba(255, 215, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence'
    ))
    
    subtitle_text = f"MAE: {metrics_dict['MAE']:.1f} | MAPE: {metrics_dict['MAPE']:.1f}%<br>Forecast simulated on historical data window only"
    
    fig.update_layout(title=f"Forecast: {metric} - {country_name}<br><sup>{subtitle_text}</sup>")
    return apply_custom_theme(fig)
