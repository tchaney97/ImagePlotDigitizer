import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

def color_to_value(color, colormap, vmin=1, vmax=1e3, num_samples=1000, log=True):
    """
    Map a given color to a data value based on the provided colormap.

    Args:
    - color: The input color as a tuple of (R, G, B) in the range [0, 255].
    - colormap: The colormap used for mapping.
    - vmin: The minimum data value corresponding to the colormap.
    - vmax: The maximum data value corresponding to the colormap.
    - num_samples: Number of samples to use for finding the closest match.

    Returns:
    - value: The data value corresponding to the input color.
    """
    # Normalize the color from [0, 255] to [0, 1]
    normalized_color = np.array(color) / 255.0

    # Normalize the data range
    if log:
        norm = matplotlib.colors.LogNorm(vmin, vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin, vmax)
    
    # Create a scalar mappable object
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=colormap)

    # Sample the colormap
    if log:
        sample_values = np.logspace(np.log10(vmin), np.log10(vmax), num_samples)
    else:
        sample_values = np.linspace(vmin, vmax, num_samples)
    sample_colors = scalar_map.to_rgba(sample_values)[:, :3]  # Ignore the alpha channel

    # Calculate the distance between the input color and sampled colors
    distances = np.sqrt(np.sum((sample_colors - normalized_color) ** 2, axis=1))

    # Find the index of the closest matching color
    closest_index = np.argmin(distances)

    # Get the corresponding data value
    value = sample_values[closest_index]

    return value

def generate_x_axis_map(tuple1, tuple2, img, log=False):
    """
    Generate a 1D x-axis array mapping each pixel column to x-axis values.

    Args:
    - tuple1: A tuple (col1, x1) where col1 is a column number and x1 is the corresponding x-axis value.
    - tuple2: A tuple (col2, x2) where col2 is a column number and x2 is the corresponding x-axis value.
    - img: image array of values
    - log: A flag indicating whether the x-axis is logarithmic (True) or linear (False).

    Returns:
    - x_axis_array: A 1D array mapping each pixel column to x-axis values.
    """
    col1, x1 = tuple1
    col2, x2 = tuple2
    num_columns = img.shape[1]

    if log:
        # Logarithmic scale interpolation
        log_x1 = np.log10(x1)
        log_x2 = np.log10(x2)
        spacing = (log_x2-log_x1)/(col2-col1)
        start_x = x1-(col1*spacing)
        stop_x = x1+((num_columns-col1)*spacing)
        log_x_axis_array = np.linspace(start_x, stop_x, num_columns)
        x_axis_array = np.power(10, log_x_axis_array)
    else:
        # Linear scale interpolation
        spacing = (x2-x1)/(col2-col1)
        start_x = x1-(col1*spacing)
        stop_x = x1+((num_columns-col1)*spacing)
        x_axis_array = np.linspace(start_x, stop_x, num_columns)

    return x_axis_array

def generate_y_axis_map(tuple1, tuple2, img, log=False):
    """
    Generate a 1D y-axis array mapping each pixel column to x-axis values.

    Args:
    - tuple1: A tuple (row1, y1) where row1 is a column number and y1 is the corresponding y-axis value.
    - tuple2: A tuple (row2, y2) where row2 is a column number and y2 is the corresponding y-axis value.
    - img: image array of values
    - log: A flag indicating whether the y-axis is logarithmic (True) or linear (False).

    Returns:
    - y_axis_array: A 1D array mapping each pixel row to y-axis values.
    """
    row1, y1 = tuple1
    row2, y2 = tuple2
    num_rows = img.shape[0]

    if log:
        # Logarithmic scale interpolation
        log_y1 = np.log10(y1)
        log_y2 = np.log10(y2)
        spacing = (log_y2-log_y1)/(row2-row1)
        start_y = y1-(row1*spacing)
        stop_y = y1+((num_rows-row1)*spacing)
        log_y_axis_array = np.linspace(start_y, stop_y, num_rows)
        y_axis_array = np.power(10, log_y_axis_array)
    else:
        # Linear scale interpolation
        spacing = (y2-y1)/(row2-row1)
        start_y = y1-(row1*spacing)
        stop_y = y1+((num_rows-row1)*spacing)
        y_axis_array = np.linspace(start_y, stop_y, num_rows)

    return y_axis_array

def crop_image(img, x_vals, y_vals, x_axis, y_axis):
    """
    Crops image based on defined x and y values.

    Args:
    - img: image array
    - x_vals: A tuple (x1, x2) of x-values that you would like included in the cropped image
    - y_vals: A tuple (y1, y2) of y-values that you would like included in the cropped image
    - x_axis: A 1D array mapping each pixel column to x-axis values.
    - y_axis: A 1D array mapping each pixel row to y-axis values.

    Returns:
    - cropped_img: The cropped image array.
    - cropped_x_axis: A 1D array mapping each pixel column to x-axis values.
    - cropped_y_axis: A 1D array mapping each pixel row to y-axis values.
    """
    x1, x2 = x_vals
    y1, y2 = y_vals

    # Find the indices corresponding to the x and y values
    x_start_idx = np.argmin(np.abs(x_axis-x1))
    x_end_idx = np.argmin(np.abs(x_axis-x2))
    y_start_idx = np.argmin(np.abs(y_axis-y1))
    y_end_idx = np.argmin(np.abs(y_axis-y2))

    if x_end_idx<x_start_idx:
        x_end_idx, x_start_idx = x_start_idx, x_end_idx
    if y_end_idx<y_start_idx:
        y_end_idx, y_start_idx = y_start_idx, y_end_idx
    # Crop the image and the axis arrays
    cropped_img = img[y_start_idx:y_end_idx, x_start_idx:x_end_idx]
    cropped_x_axis = x_axis[x_start_idx:x_end_idx]
    cropped_y_axis = y_axis[y_start_idx:y_end_idx]

    return cropped_img, cropped_x_axis, cropped_y_axis