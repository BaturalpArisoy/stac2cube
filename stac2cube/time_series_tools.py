import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.animation as animation
import matplotlib.animation as manimation
import ipywidgets as widgets
from IPython.display import display
from matplotlib.ticker import ScalarFormatter
from PIL import Image

#####################
# GENERATE ANIMATION
#####################

def generate_animation(stac, export_folder, display_mode, frame_interval_ms=1000):
    """
    Create figures for each time slice and export a GIF animation.
    After the GIF is created, all figures in the figures folder are deleted.
    
    Parameters:
      stac (xarray.DataArray): Data array with dimensions (time, band, y, x).
      export_folder (str): Folder to save figures and animation.
      display_mode (str): One of 'rgb', 'ndvi', or 'ndwi'.
      frame_interval_ms (int): Delay between frames in milliseconds.
                               Smaller = faster animation. Default is 1000 ms.
    """
    if display_mode == 'rgb':
        image_stac = stac.sel(band=["red", "green", "blue"])
        vmin = None
        vmax = None
    elif display_mode in ['ndvi', 'ndwi']:
        image_stac = stac.sel(band=display_mode)
        # Global limits over entire time series for consistent colour scale
        vmin, vmax = _get_global_limits(image_stac)
    else:
        raise ValueError(f"Unknown display_mode: {display_mode}")

    _export_figures(image_stac, export_folder, display_mode, vmin=vmin, vmax=vmax)
    _export_animation(export_folder, display_mode, frame_interval_ms)
    _cleanup_figures(export_folder)
    
#############################
# INTERACTIVE TIME VIEW
#############################

def interactive_time_view(stac, display_mode, widget_type='slider'):
    """
    Displays an interactive time-series view of Sentinel-2 data.
    Choose between a slider and a dropdown widget for selecting the time slice.
    
    Parameters:
      stac (xarray.DataArray): The data array with dimensions (time, band, y, x).
      display_mode (str): One of 'rgb', 'ndvi', or 'ndwi'.
      widget_type (str): 'slider' for a slider widget or 'dropdown' for a dropdown widget.
    """
    if display_mode == 'rgb':
        data_stac = stac.sel(band=["red", "green", "blue"])
        vmin = None
        vmax = None
    elif display_mode in ['ndvi', 'ndwi']:
        data_stac = stac.sel(band=display_mode)
        # Use same idea: global limits for consistent colour mapping
        vmin, vmax = _get_global_limits(data_stac)
    else:
        raise ValueError(f"Unknown display_mode: {display_mode}")
        
    time_values = pd.to_datetime(data_stac.time.values)
    num_time_slices = data_stac.time.size

    def plot_time_slice(index):
        fig, ax = plt.subplots(figsize=(8, 8))
        title_str = time_values[index].strftime('%d-%m-%Y')
        
        if display_mode == 'rgb':
            rgb_values = data_stac.isel(time=index).transpose('y', 'x', 'band').values
            red_n = _normalize(rgb_values[:, :, 0])
            green_n = _normalize(rgb_values[:, :, 1])
            blue_n = _normalize(rgb_values[:, :, 2])
            rgb_image = np.dstack((red_n, green_n, blue_n))
            
            if _is_image_missing(rgb_image):
                ax.text(0.5, 0.5, 'Missing Data', fontsize=18,
                        ha='center', va='center', transform=ax.transAxes)
            else:
                ax.imshow(
                    rgb_image,
                    interpolation="bicubic",
                    extent=[data_stac.x.min(), data_stac.x.max(),
                            data_stac.y.min(), data_stac.y.max()]
                )
        elif display_mode in ['ndvi', 'ndwi']:
            data = data_stac.isel(time=index).values
            cmap = "RdYlGn" if display_mode == 'ndvi' else "Blues"
            title_str += " (NDVI)" if display_mode == 'ndvi' else " (NDWI)"
            ax.imshow(
                data, cmap=cmap,
                extent=[data_stac.x.min(), data_stac.x.max(),
                        data_stac.y.min(), data_stac.y.max()],
                vmin=vmin, vmax=vmax
            )
        else:
            ax.text(0.5, 0.5, 'Unknown display_mode', fontsize=18,
                    ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(title_str, fontsize=14)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        plt.show()
    
    if widget_type == 'slider':
        slider_layout = widgets.Layout(width='800px')
        time_widget = widgets.IntSlider(
            min=0, 
            max=num_time_slices-1, 
            step=1, 
            value=0, 
            description='Time', 
            layout=slider_layout
        )
    elif widget_type == 'dropdown':
        options = [(t.strftime('%d-%m-%Y'), i) for i, t in enumerate(time_values)]
        time_widget = widgets.Dropdown(
            options=options,
            value=0,
            description='Date:',
            disabled=False,
            layout=widgets.Layout(width='300px')
        )
    else:
        raise ValueError(f"Unknown widget_type: {widget_type}. Use 'slider' or 'dropdown'.")
    
    widgets.interact(plot_time_slice, index=time_widget)

#############################
# HELPER FUNCTIONS
#############################

def _export_figures(stac, export_folder, display_mode, vmin=None, vmax=None):
    output_dir = os.path.join(export_folder, 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_time_slices = stac.time.size
    time_values = pd.to_datetime(stac.time.values)

    # If NDVI/NDWI and no limits provided, compute here as fallback
    if display_mode in ['ndvi', 'ndwi'] and (vmin is None or vmax is None):
        vmin, vmax = _get_global_limits(stac)

    for i in range(num_time_slices):
        fig, ax = plt.subplots(figsize=(8, 8))
        title_str = time_values[i].strftime('%d-%m-%Y')
        
        if display_mode == 'rgb':
            rgb_values = stac.isel(time=i).transpose('y', 'x', 'band').values
            red_n = _normalize(rgb_values[:, :, 0])
            green_n = _normalize(rgb_values[:, :, 1])
            blue_n = _normalize(rgb_values[:, :, 2])
            rgb_image = np.dstack((red_n, green_n, blue_n))
            
            if _is_image_missing(rgb_image):
                print(f"Skipping plot for {time_values[i].strftime('%Y%m%d_%H%M%S')} due to missing data.")
                plt.close()
                continue

            ax.imshow(
                rgb_image,
                interpolation="bicubic", 
                extent=[stac.x.min(), stac.x.max(),
                        stac.y.min(), stac.y.max()]
            )
        elif display_mode in ['ndvi', 'ndwi']:
            data = stac.isel(time=i).values
            if display_mode == 'ndvi':
                cmap = "RdYlGn"
                title_str += " (NDVI)"
            else:
                cmap = "Blues"
                title_str += " (NDWI)"
            ax.imshow(
                data, cmap=cmap, 
                extent=[stac.x.min(), stac.x.max(),
                        stac.y.min(), stac.y.max()], 
                vmin=vmin, vmax=vmax
            )
        else:
            print(f"Unknown display_mode: {display_mode}. Skipping time slice {i}.")
            plt.close()
            continue

        ax.set_title(title_str, fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)

        timestamp = time_values[i].strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/{display_mode.upper()}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=100)
        plt.close()
    
def _export_animation(export_folder, display_mode, frame_interval_ms=1000):
    if frame_interval_ms <= 0:
        raise ValueError("frame_interval_ms must be a positive integer.")

    output_dir = os.path.join(export_folder, 'figures')
    gif_output_path = os.path.join(export_folder, 'animations', f'animated_{display_mode}.gif')

    anim_output_dir = os.path.dirname(gif_output_path)
    if not os.path.exists(anim_output_dir):
        os.makedirs(anim_output_dir)
        
    files = sorted(glob.glob(f"{output_dir}/*.png"))
    image_array = []
    for my_file in files:
        image = Image.open(my_file)
        image_array.append(image)

    print('Number of images:', len(image_array))
    
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(image_array[0], animated=True)

    def update(i):
        im.set_array(image_array[i])
        return im, 

    # Derive fps from interval (avoid fps=0)
    fps = max(1, int(1000 / frame_interval_ms))

    animation_fig = manimation.FuncAnimation(
        fig,
        update,
        frames=len(image_array),
        interval=frame_interval_ms,
        blit=True,
        repeat_delay=2000,
    )
    animation_fig.save(gif_output_path, writer='imagemagick', fps=fps, dpi=300)

def _cleanup_figures(export_folder):
    """
    Delete all PNG files in the figures folder.
    """
    figures_dir = os.path.join(export_folder, 'figures')
    files = glob.glob(os.path.join(figures_dir, '*.png'))
    for file_path in files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def _normalize(band, clip_percentile=2):
    band_min, band_max = np.nanpercentile(band, [clip_percentile, 100 - clip_percentile])
    if band_max == band_min:
        return np.zeros_like(band)
    band = np.clip(band, band_min, band_max)
    normalized = (band - band_min) / (band_max - band_min)
    normalized = np.nan_to_num(normalized, nan=0.5)
    return normalized

def _is_image_missing(rgb_image, threshold=0.1):
    black_pixels = np.sum(np.all(rgb_image == 0, axis=2))
    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
    return (black_pixels / total_pixels) > threshold

def _get_global_limits(stac, lower_percentile=2, upper_percentile=98):
    """
    Compute global vmin/vmax for NDVI/NDWI over the whole time series.
    Uses percentiles to be robust against outliers.
    """
    vals = stac.values  # expected shape: (time, y, x)
    vmin = float(np.nanpercentile(vals, lower_percentile))
    vmax = float(np.nanpercentile(vals, upper_percentile))

    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    return vmin, vmax
