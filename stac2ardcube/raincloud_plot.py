#!/usr/bin/env python3
"""
raincloud_plot.py

This module provides a function to generate raincloud plots from a NetCDF dataset.
The main function, `generate_raincloud_plots()`, accepts:
    - input_path: Path to the input NetCDF file (including filename)
    - output_dir: Directory where the output plots will be saved (the file names are generated automatically)
    - bands: A list of band names to extract from the dataset (e.g., ['ndvi', 'ndwi'])
    - stats: A list of statistics to compute (e.g., ['mean', 'median', 'max'])

For each combination of band and stat, the function will generate a separate raincloud plot.
The output file name is generated as: f"{band}_{stat}.png"
"""

import os
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt  # Ensure ptitprince is installed in your environment

def _load_dataset(input_path, band):
    """
    Load the dataset from the given path and select the specified band.

    Parameters:
        input_path (str): Path to the NetCDF file.
        band (str): Band name to select from the dataset.

    Returns:
        xarray.DataArray: The data array corresponding to the specified band.
    """
    ds = xr.open_dataset(input_path)
    da = ds.Spectral_Temporal_Stack
    return da.sel(band=band)

def _group_by_year(da_band):
    """
    Group the data by year based on the 'time' coordinate.

    Parameters:
        da_band (xarray.DataArray): The data array with the specified band selected.

    Returns:
        dict: A dictionary with years as keys and the corresponding data arrays as values.
    """
    return {year: group for year, group in da_band.groupby('time.year')}

def _compute_series_list(data_by_year, stat):
    """
    Compute the specified statistic for each year across spatial dimensions ("y", "x")
    and return a list of pandas Series.

    Parameters:
        data_by_year (dict): Dictionary with years as keys and data arrays as values.
        stat (str): The name of the statistic to compute (e.g., 'mean', 'max').

    Returns:
        list: A list of pandas Series, each corresponding to a year's computed statistic.
    """
    series_list = []
    for year, group in sorted(data_by_year.items()):
        # Dynamically compute the statistic using getattr()
        computed = getattr(group, stat)(dim=("y", "x"))
        series = pd.Series(computed.values, name=str(year))
        series_list.append(series.reset_index(drop=True))
    return series_list

def _create_dataframe(series_list):
    """
    Combine all the series into a DataFrame and convert it to long format.

    Parameters:
        series_list (list): List of pandas Series.

    Returns:
        pandas.DataFrame: A melted DataFrame with 'Period' and 'Value' columns.
    """
    df = pd.concat(series_list, axis=1)
    # Melt the DataFrame to long format.
    df_melted = df.melt(var_name='Period', value_name='Value').dropna().reset_index(drop=True)
    return df_melted

def _create_raincloud_plot(df_melted, band, stat):
    """
    Create and format the raincloud plot using ptitprince and seaborn.

    Parameters:
        df_melted (pandas.DataFrame): The melted DataFrame with the computed statistics.
        band (str): The band name (used in plot titles and labels).
        stat (str): The statistic computed (used in plot titles and labels).

    Returns:
        tuple: A tuple containing the matplotlib Figure and Axes objects.
    """
    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Create the raincloud plot.
    pt.RainCloud(x='Period', y='Value', data=df_melted, orient='v',
                 width_viol=0.6, point_size=2, palette="Set2", ax=ax)
    
    # Update x-axis labels to include sample counts per year.
    counts = df_melted['Period'].value_counts()
    new_labels = [f"{p} ({counts[p]})" if p in counts.index else p for p in df_melted['Period'].unique()]
    ax.set_xticklabels(new_labels)
    
    # Set titles and labels.
    ax.set_title(f"Distribution of {stat.capitalize()} {band.upper()} Values Across Different Periods",
                 fontsize=20, pad=30)
    ax.set_xlabel("Time Period", fontsize=16, labelpad=20)
    ax.set_ylabel(f"{stat.capitalize()} {band.upper()}", fontsize=16, labelpad=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Adjust layout to prevent crowding.
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    return fig, ax

def generate_raincloud_plots(input_path, output_dir, bands, stats):
    """
    Generate raincloud plots from the given NetCDF dataset for each combination of band and statistic.

    Parameters:
        input_path (str): Path to the input NetCDF file (including filename).
        output_dir (str): Directory to save the output plots (the file names are generated automatically).
        bands (list): List of band names to extract from the dataset (e.g., ['ndvi', 'ndwi']).
        stats (list): List of statistics to compute (e.g., ['mean', 'median', 'max']).

    For each combination of band and stat, the function:
      - Loads the dataset and selects the specified band.
      - Groups the data by year.
      - Computes the desired statistic across spatial dimensions.
      - Creates a melted DataFrame for plotting.
      - Generates, displays, and saves a raincloud plot to the specified directory.
        The file name is automatically generated as: f"{band}_{stat}.png"
    """
    # Ensure bands and stats are lists; if not, convert them.
    if not isinstance(bands, list):
        bands = [bands]
    if not isinstance(stats, list):
        stats = [stats]
    
    for band in bands:
        # Load the data and select the desired band.
        da_band = _load_dataset(input_path, band)
        # Group the data by year.
        data_by_year = _group_by_year(da_band)
        
        # (Optional) Print the shape of the data for each year for verification.
        #for year, group in sorted(data_by_year.items()):
            #print(f"Band: {band} | Year: {year}, data shape: {group.shape}")
        
        for stat in stats:
            # Compute the statistic for each year and create a list of Series.
            series_list = _compute_series_list(data_by_year, stat)
            
            # Combine the series into a single DataFrame and melt it to long format.
            df_melted = _create_dataframe(series_list)
            #print(f"Band: {band} | Stat: {stat}")
            #print(df_melted)
            
            # Create the raincloud plot.
            fig, ax = _create_raincloud_plot(df_melted, band, stat)
            
            # Construct the output file name and full path.
            filename = f"{band}_{stat}.png"
            full_output_path = os.path.join(output_dir, filename)
            
            # Save the figure as a high-quality PNG.
            fig.savefig(full_output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {full_output_path}")
            
            # Display the plot.
            #plt.show()
            # Close the figure to free up memory.
            #plt.close(fig)

if __name__ == '__main__':
    # Example usage:
    input_path = "/dss/dsstbyfs02/pr94no/pr94no-dss-0001/drylands/results/orog_2.nc"
    # Provide the output directory (without file name)
    output_dir = "/dss/dsstbyfs02/pr94no/pr94no-dss-0001/drylands/graphs/orog_2"
    
    # Example: process two bands and two stats.
    bands = ["ndwi", "ndvi"]
    stats = ["mean", "median"]
    
    generate_raincloud_plots(input_path, output_dir, bands, stats)
