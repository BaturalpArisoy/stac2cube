import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.animation as animation
import matplotlib.animation as manimation
import glob
from matplotlib.ticker import ScalarFormatter
from PIL import Image



def generate_animation(stac, export_folder, animation):
    if animation == 'rgb':
        image_stac = stac.sel(band=["red", "green", "blue"])
    elif animation in ['ndvi', 'ndwi']:
        image_stac = stac.sel(band=animation)
    else:
        raise ValueError(f"Unknown animation type: {animation}")

    _export_figures(image_stac, export_folder, animation)
    _export_animation(export_folder, animation)
    
    
def _export_animation(export_folder, animation):
    output_dir = os.path.join(export_folder, 'figures')
    gif_output_path = os.path.join(export_folder, 'animations', f'animated_{animation}.gif')

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

    animation_fig = manimation.FuncAnimation(fig, update, frames=len(image_array), interval=1000, blit=True, repeat_delay=2000)
    animation_fig.save(gif_output_path, writer='imagemagick', fps=60, dpi=300)
    
    
def _export_figures(stac, export_folder, animation):
    output_dir = os.path.join(export_folder, 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_time_slices = stac.time.size
    time_values = pd.to_datetime(stac.time.values)

    for i in range(num_time_slices):
        fig, ax = plt.subplots(figsize=(8, 8))
        title_str = time_values[i].strftime('%d-%m-%Y')
        
        if animation == 'rgb':
            rgb_values = stac.isel(time=i).transpose('y', 'x', 'band').values
            rgb_n = _normalize(rgb_values)
            gamma = _calculate_gamma(rgb_n)
            red_n = _normalize(rgb_values[:, :, 0])
            green_n = _normalize(rgb_values[:, :, 1])
            blue_n = _normalize(rgb_values[:, :, 2])
            red_g = _gammacorr(red_n, gamma)
            green_g = _gammacorr(green_n, gamma)
            blue_g = _gammacorr(blue_n, gamma)
            rgb_image = np.dstack((red_g, green_g, blue_g))
            
            if _is_image_missing(rgb_image):
                print(f"Skipping plot for {time_values[i].strftime('%Y%m%d_%H%M%S')} due to missing data.")
                plt.close()
                continue

            ax.imshow(rgb_image, interpolation="bicubic", 
                      extent=[stac.x.min(), stac.x.max(), stac.y.min(), stac.y.max()])
        elif animation in ['ndvi', 'ndwi']:
            # For ndvi/ndwi, the dataset is already selected with the appropriate band.
            # The data is assumed to be 2D per time slice.
            data = stac.isel(time=i).values
            if animation == 'ndvi':
                cmap = "RdYlGn"
                title_str += " (NDVI)"
            else:  # ndwi
                cmap = "Blues"
                title_str += " (NDWI)"
            ax.imshow(data, cmap=cmap, 
                      extent=[stac.x.min(), stac.x.max(), stac.y.min(), stac.y.max()], 
                      vmin=-1, vmax=1)
        else:
            print(f"Unknown animation type: {animation}. Skipping time slice {i}.")
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
        filename = f"{output_dir}/{animation.upper()}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=100)
        plt.close()
    
    
def _is_image_missing(rgb_image, threshold=0.1):
    black_pixels = np.sum(np.all(rgb_image == 0, axis=2))
    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
    return (black_pixels / total_pixels) > threshold


def _normalize(band):
    band_min, band_max = band.min(), band.max()
    return (band - band_min) / (band_max - band_min)


def _gammacorr(band, gamma):
    return np.power(band, 1/gamma)


def _calculate_gamma(rgb_image):
    brightness = np.mean(rgb_image, axis=2)
    median_brightness = np.median(brightness)
    std_brightness = np.std(brightness)
    if median_brightness < 0.5:
        gamma = 2.0 + (0.5 - median_brightness) * 1.5
    else:
        gamma = 1.0 / (1.0 + (median_brightness - 0.5) * 1.5)
    return gamma