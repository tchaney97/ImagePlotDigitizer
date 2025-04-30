import numpy as np
from scipy.interpolate import interpn
from scipy.spatial import KDTree
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

def colormap_from_image_line(img_rgb, index, start, stop, width=1, normalize=True, orientation='vertical', flip=False, name='custom cmap'):
    """
    Extract colors along a (sub-)line from an RGB(A) image and create
    a ListedColormap from those samples.

    Parameters
    ----------
    img_rgb : ndarray, shape (H, W, 3) or (H, W, 4)
        Input image, uint8 in [0–255] or float in [0–1].
    index : int
        Row (if horizontal) or column (if vertical) to sample.
    start : int or None, default None
        Starting column (if horizontal) or row (if vertical).  
        If None, begins at 0.
    stop : int or None, default None
        Ending column (if horizontal) or row (if vertical), exclusive.
    width : int, default=1
        number of pixels to average. will average over nearest odd value number
        If None, goes to image boundary.
    normalize : bool, default True
        If True, convert uint8→float by dividing by 255 (or scale floats >1 down).
    orientation : {'horizontal', 'vertical'} default 'vertical'
        'horizontal' → grab row `index`; 'vertical' → grab column `index`.
    flip : bool default 'False'
        If True, flips colormap order.
    name : str, default "extracted_cmap"
        Name for the resulting colormap.
    
    Returns
    -------
    cmap : ListedColormap
        A matplotlib colormap whose colors are the sampled pixels in order.
    """
    
    h, w = img_rgb.shape[:2]
    # clamp start/stop
    if start is None:
        start = 0
    if stop is None:
        if orientation == "horizontal":
            stop=w
        else:
            stop=h
            

    # slice out the line of pixels
    if orientation == "horizontal":
        # limit columns from start to stop
        line = np.mean(img_rgb[index-width//2:index+1+width//2, start:stop, :], axis=0)
    elif orientation == "vertical":
        # limit rows from start to stop
        line = np.mean(img_rgb[start:stop, index-width//2:index+1+width//2, :], axis=1)
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    
    # convert to float and optionally normalize
    colors = line.astype(float)
    if normalize and colors.max() > 1.0:
        colors /= 255.0
    
    # drop alpha channel if present
    if colors.shape[-1] == 4:
        colors = colors[:, :3]
        
    if flip:
        colors = np.flip(colors, axis=0)
    
    # build a ListedColormap
    cmap = ListedColormap(colors, name=name)

    return cmap
    
def color_to_value(img_rgb, colormap, min_val=1, max_val=1e3, num_samples=256, log=True):
    """
    Map a given color to a data value based on the provided colormap.

    Args:
    - img_rgb: The input image as an RBG array with dimensions (h,w,3)
      and values in the range [0, 255].
    - colormap: The matplotlib colormap used for mapping.
    - min_val: The minimum data value corresponding to the colormap.
    - max_val: The maximum data value corresponding to the colormap.
    - num_samples: Number of samples to use for finding the closest match.
    - log: boolean for log-scaled image or linear scaled image

    Returns:
    - img_vals: The data value corresponding to the colors in img_rgb.
    """
    h,w,_ = np.shape(img_rgb)
    
    sample_positions = np.linspace(0,1,num_samples)
    sample_colors = colormap(sample_positions)[:,:3]
    if log:
        colormap_values = np.logspace(np.log10(min_val), np.log10(max_val), num_samples)
    else:
        colormap_values = np.linspace(min_val, max_val, num_samples)
    # -1 in shape infers dimension
    flat_img = img_rgb.reshape(-1,3).astype(float)
    
    #convert from 0-255 rgb to 0-1 rgb
    flat_img /= 255

    # Use KDTree for efficient matching of colors
    tree = KDTree(sample_colors)
    _, closest_idx = tree.query(flat_img)
    
    values = colormap_values[closest_idx]
    img_vals = values.reshape(h, w)

    return img_vals

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

def generate_arb_x_axis_map(indices, coordinates, img):
    """
    Generate a 1-D x-axis array mapping each pixel index to x-axis coordinate value: 
    - For nonlinear, arbitrarily spaced axes

    Args:
    - indices: list of x-axis column numbers to extract
    - coordinates: corresponding list of x-axis values at the specified column numbers (order matters)
    - img: image array of values
    """

    # Generate full x-axis based on linear spacing
    spacing = (coordinates[-1]-coordinates[0])/(indices[-1]-indices[0])
    start_x = coordinates[0]-(indices[0]*spacing)
    stop_x = coordinates[0]+((img.shape[1]-indices[0])*spacing)
    x_axis_array = np.linspace(start_x, stop_x, img.shape[1])
    
    # Overwrite with linearly spaced segments between specificed indices & coordinates:
    for (col1, col2), (x1, x2) in zip(zip(indices[:-1], indices[1:]), zip(coordinates[:-1], coordinates[1:])):
        if col2<col1:
            step=-1
        else:
            step=1
        x_axis_array[col1:col2+step:step] = np.linspace(x1, x2, num=np.abs(col2-col1)+1)

    return x_axis_array

def generate_arb_y_axis_map(indices, coordinates, img):
    """
    Generate a 1-D y-axis array mapping each pixel index to x-axis coordinate value: 
    - For nonlinear, arbitrarily spaced axes

    Args:
    - indices: list of y-axis column numbers to extract
    - coordinates: corresponding list of y-axis values at the specified column numbers (order matters)
    - img: image array of values
    """

    # Generate full x-axis based on linear spacing
    spacing = (coordinates[-1]-coordinates[0])/(indices[-1]-indices[0])
    start_y = coordinates[0]-(indices[0]*spacing)
    stop_y = coordinates[0]+((img.shape[0]-indices[0])*spacing)
    y_axis_array = np.linspace(start_y, stop_y, img.shape[0])
    
    # Overwrite with linearly spaced segments between specificed indices & coordinates:
    for (row1, row2), (y1, y2) in zip(zip(indices[:-1], indices[1:]), zip(coordinates[:-1], coordinates[1:])):
        if row2<row1:
            step=-1
        else:
            step=1
        y_axis_array[row1:row2+step:step] = np.linspace(y1, y2, num=np.abs(row2-row1)+1)  

    return y_axis_array

def make_axes_linear(img, x_axis, y_axis):
    """
    Converts an image composed of pixels corresponding to arbitrarily spaced axes to an 
    image composed of pixels corresponding to linearly spaced axes

    Args:
    - img: image array
    - x_axis: A 1D array mapping each pixel column to x-axis values.
    - y_axis: A 1D array mapping each pixel row to y-axis values.

    Returns:
    - cropped_img: The cropped image array.
    - cropped_x_axis: A 1D array mapping each pixel column to x-axis values.
    - cropped_y_axis: A 1D array mapping each pixel row to y-axis values.
    """
    x_axis = np.asarray(x_axis)
    y_axis = np.asarray(y_axis)

    #flip any axis that is in descending order
    flip_x = False
    flip_y = False

    if x_axis[1] < x_axis[0]:
        x_axis = x_axis[::-1]
        img = img[:, ::-1]
        flip_x = True

    if y_axis[1] < y_axis[0]:
        y_axis = y_axis[::-1]
        img = img[::-1, :]
        flip_y = True

    
    old_points = (y_axis, x_axis)
    new_x_axis = np.linspace(np.min(x_axis), np.max(x_axis), len(x_axis))
    new_y_axis = np.linspace(np.min(y_axis), np.max(y_axis), len(y_axis))
    #flip y-axis here to maintain img array convention
    Xn, Yn = np.meshgrid(new_x_axis, new_y_axis, indexing='xy')
    new_points = np.vstack([Yn.ravel(), Xn.ravel()]).T
    new_img = interpn(old_points, img, new_points)
    new_img = new_img.reshape(Xn.shape)
    # new_img = np.flipud(new_img)
    # new_y_axis = np.flip(new_y_axis)

    if flip_y:
        new_img = np.flipud(new_img)
        new_y_axis = new_y_axis[::-1]
    if flip_x:
        new_img = new_img[:, ::-1]
        new_x_axis = new_x_axis[::-1]

    return new_img, new_x_axis, new_y_axis

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