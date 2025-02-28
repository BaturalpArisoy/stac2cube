import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.animation as animation
import glob
from matplotlib.ticker import ScalarFormatter
from PIL import Image



def generate_animation(stac):
    
    #stac = stac[['red', 'green', 'blue']]
    
    #rgb = xr.concat([stac.red, stac.green, stac.blue], dim='band')
    #rgb = rgb.assign_coords(band=['red', 'green', 'blue'])
    #rgb = rgb.transpose('time', 'band', 'y', 'x')
    
    rgb = stac.sel(band=["red","green","blue"])
    
    _export_figures(rgb)
    
    
    _export_animation()
    
    
    
    
    
def _export_animation():
    
    home_dir = os.path.expanduser('~')
    
    output_dir = os.path.join(home_dir, 'terrabyte_cube/interactive/figures')
    gif_output_path = os.path.join(home_dir, 'terrabyte_cube/interactive/animations/animated_RGB.gif')

#    output_dir = './figures'
#    gif_output_path = "./animations/animated_RGB.gif"
    anim_output_dir = os.path.dirname(gif_output_path)
    if not os.path.exists(anim_output_dir):
        os.makedirs(anim_output_dir)
    # Get all PNG files and sort them by their filenames
    files = sorted(glob.glob(f"{output_dir}/*.png"))

    image_array = []

    for my_file in files:
        image = Image.open(my_file)
        image_array.append(image)

    print('Number of images:', len(image_array))

    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Hide the axes (remove local coordinate system)
    ax.axis('off')

    # Set the initial image
    im = ax.imshow(image_array[0], animated=True)

    def update(i):
        im.set_array(image_array[i])
        return im, 

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=1000, blit=True, repeat_delay=2000)

    # Show the animation
    #plt.show()

    # Save the animation with high quality
    animation_fig.save(gif_output_path, writer='imagemagick', fps=60, dpi=300)
    
    
def _export_figures(stac):
    
    home_dir = os.path.expanduser('~')
    output_dir = os.path.join(home_dir, 'terrabyte_cube/interactive/figures')
    #output_dir = './figures' # change this later!

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_time_slices = stac.time.size
    time_values = pd.to_datetime(stac.time.values)

    def format_tick(value, pos):
        # Ensure scientific notation with 2 decimal places
        return f'{value:.2e}' if value != 0 else '0.00'



    for i in range(num_time_slices):
        fig, ax = plt.subplots(figsize=(8, 8))

        rgb_values = stac.isel(time=i).transpose('y', 'x', 'band').values

        rgb_n= _normalize(rgb_values)

        gamma = _calculate_gamma(rgb_n)
        
        red_n= _normalize(rgb_values[:, :, 0])
        green_n= _normalize(rgb_values[:, :, 1])
        blue_n= _normalize(rgb_values[:, :, 2])

        red_g = _gammacorr(red_n, gamma)
        green_g = _gammacorr(green_n, gamma)
        blue_g = _gammacorr(blue_n, gamma)

        rgb_image = np.dstack((red_g, green_g, blue_g))

        if _is_image_missing(rgb_image):
            print(f"Skipping plot for {time_values[i].strftime('%Y%m%d_%H%M%S')} due to missing data.")
            plt.close()  # Close the plot to free up memory
            continue


        # Display the RGB image
        ax.imshow(rgb_image, interpolation="bicubic", extent=[stac.x.min(), stac.x.max(),
                                 stac.y.min(), stac.y.max()])
        ax.set_title(time_values[i].strftime('%d-%m-%Y'), fontsize=14)  # Use the actual date as the title

        ax.tick_params(axis='x', rotation=45)
        #ax.set_xticklabels(ax.get_xticks(), rotation=45)

        # Ensure scientific notation on both axes
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)

        # Generate a timestamp string for the filename
        timestamp = time_values[i].strftime('%Y%m%d_%H%M%S')
        # Create the filename with the plot title and timestamp
        filename = f"{output_dir}/RGB_{timestamp}.png"

        plt.tight_layout()
        plt.savefig(filename, dpi=100) #300
        plt.close()  # Close the plot to free up memory
        #print(f"Plot saved as {filename}")
    
    
      
    
def _is_image_missing(rgb_image, threshold=0.1):
    """
    Determine if the image is considered 'missing' based on the proportion of black pixels.
    :param rgb_image: 3D numpy array (H x W x 3)
    :param threshold: proportion threshold to consider an image as missing (default 50%)
    :return: True if the image is considered missing, False otherwise
    """
    black_pixels = np.sum(np.all(rgb_image == 0, axis=2))
    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
    black_pixel_proportion = black_pixels / total_pixels
    return black_pixel_proportion > threshold


def _normalize(band):
    band_min, band_max = band.min(), band.max()
    return (band - band_min) / (band_max - band_min)


def _gammacorr(band, gamma):
    return np.power(band, 1/gamma)


def _calculate_gamma(rgb_image):
    brightness = np.mean(rgb_image, axis=2)
    
    # Calculate the median and standard deviation of the brightness
    median_brightness = np.median(brightness)
    std_brightness = np.std(brightness)
    
    # Adjust gamma based on median brightness
    if median_brightness < 0.5:
        # Increase gamma as brightness decreases
        gamma = 2.0 + (0.5 - median_brightness) * 1.5
        
    else:
        # Decrease gamma as brightness increases
        gamma = 1.0 / (1.0 + (median_brightness - 0.5) * 1.5)
        
    return gamma