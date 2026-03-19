import pandas as pd
import streamlit as st
import numpy as np
import os

@st.cache_data
def load_and_preprocess_data():
    """
    Load data directly from Our World in Data and preprocess it.
    Filters to the specified range: Jan 2020 - May 2023.
    """
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    local_path = "owid-covid-data.csv"
    
    # Define columns to load to save memory
    usecols = [
        "iso_code", "continent", "location", "date", 
        "total_cases", "new_cases", "total_deaths", "new_deaths",
        "total_vaccinations", "people_vaccinated", "people_fully_vaccinated",
        "hosp_patients", "positive_rate", "stringency_index", "gdp_per_capita",
        "population"
    ]
    
    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, usecols=usecols, parse_dates=["date"])
        else:
            try:
                # Try GitHub mirror first if the main site is blocked by DNS/Network
                github_url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
                df = pd.read_csv(github_url, usecols=usecols, parse_dates=["date"])
            except:
                # Fallback to original url
                df = pd.read_csv(url, usecols=usecols, parse_dates=["date"])
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()
        
    # Filter dates to Jan 1, 2020 - May 31, 2023
    start_date = pd.to_datetime("2020-01-01")
    end_date = pd.to_datetime("2023-05-31")
    
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    
    # Sort by location and date
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    
    # Basic data cleaning: handle missing values with forward fill then interpolation
    cols_to_fill = [
        "total_cases", "new_cases", "total_deaths", "new_deaths",
        "total_vaccinations", "people_vaccinated", "people_fully_vaccinated",
        "hosp_patients", "positive_rate"
    ]
    
    for col in cols_to_fill:
        df[col] = df.groupby("location")[col].ffill()
        
    # Interpolate numeric columns safely (using group and apply logic cautiously around all-NaN columns)
    def safe_interpolate(group):
        if group.isna().all():
            return group
        return group.interpolate(method='linear', limit_direction='both')
    
    # Optimize interpolation by bypassing costly `.apply` when possible, but let's stick to safe fallback
    for col in cols_to_fill:
        df[col] = df.groupby("location")[col].transform(safe_interpolate)
        df[col] = df[col].fillna(0)
        
    # Proxy Active Cases: Total Cases - Total Deaths
    df["active_cases"] = df["total_cases"] - df["total_deaths"]
    df["active_cases"] = df["active_cases"].clip(lower=0)
    
    return df

@st.cache_data
def get_available_countries(df):
    """Return a list of valid countries (excluding continents/aggregates)."""
    # Exclude non-country locations like 'World', 'Europe', etc.
    countries = df[df["continent"].notna()]["location"].unique().tolist()
    return sorted(countries)
