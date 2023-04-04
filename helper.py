import numpy as np
from skimage import draw

def pix2point(depth: np.ndarray, depth_intrinsics_dict: dict) -> np.ndarray:
    """
    Convert pixel coordinates of a depth image to 3D real-world coordinates.

    Args:
        depth (np.ndarray): A 2D numpy array representing the depth image.
        depth_intrinsics_dict (dict): A dictionary containing the intrinsic parameters of the depth camera.

    Returns:
        np.ndarray: A 3D numpy array representing the real-world coordinates of the pixels in the depth image.
    """

    height, width = depth.shape

    x_s_idx, y_s_idx = np.meshgrid(np.arange(width), np.arange(height))
    point_array_x = np.zeros((height, width))
    point_array_y = np.zeros((height, width))
    x = (x_s_idx - depth_intrinsics_dict['ppx']) / depth_intrinsics_dict['fx']
    y = (y_s_idx - depth_intrinsics_dict['ppy']) / depth_intrinsics_dict['fy']
    r2 = x * x + y * y
    f = 1 + depth_intrinsics_dict['coeffs'][0] * r2 + depth_intrinsics_dict['coeffs'][
        1] * r2 * r2 + depth_intrinsics_dict['coeffs'][4] * r2 * r2 * r2
    ux = x * f + 2 * depth_intrinsics_dict['coeffs'][2] * x * y + depth_intrinsics_dict['coeffs'][3] * (r2 + 2 * x * x)
    uy = y * f + 2 * depth_intrinsics_dict['coeffs'][3] * x * y + depth_intrinsics_dict['coeffs'][2] * (r2 + 2 * y * y)
    x = ux
    y = uy
    point_array_x = depth * x
    point_array_y = depth * y
    array = np.array([point_array_x, point_array_y, depth])
    real_coorinate_depth = np.transpose(array, (1, 2, 0))

    return real_coorinate_depth


def get_food_mask(results: dict) -> tuple:
    """
    Get a mask for food items in an image and return the mask along with the indices of the food items.

    Args:
        results (dict): A dictionary containing the results of object detection on the image.

    Returns:
        tuple: A tuple containing the mask for food items in the image and the indices of the food items.
    """

    non_food_value = -1
    height = results["masks"][0].shape[0]
    width = results["masks"][0].shape[1]
    ranked_food_mask = np.ones((height, width)) * non_food_value

    sorted_idx = range(len(results["class_names"]))

    for idx in sorted_idx:
        ranked_food_mask[results["masks"][idx] == 1] = idx

    unique_indices = np.unique(ranked_food_mask).astype("int16")
    food_indices = unique_indices[unique_indices != non_food_value]

    return ranked_food_mask, food_indices

def poly2mask(vertex_row_coords: np.ndarray, vertex_col_coords: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert polygon coordinates to a binary mask.

    Parameters:
    -----------
    vertex_row_coords: list or numpy array of int
        List of y-coordinates of vertices in polygon.
    vertex_col_coords: list or numpy array of int
        List of x-coordinates of vertices in polygon.
    shape: tuple
        Shape of the image for which the mask will be created.

    Returns:
    --------
    mask: numpy array of bool
        Binary mask created from the polygon coordinates.

    """
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool_)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def json2results(data: dict, shape_: tuple) -> dict:
    """
    Convert json data to dictionary of segmentation results.

    Parameters:
    -----------
    data: dict
        Dictionary containing the segmentation data in json format.
    shape_: tuple
        Shape of the image for which the segmentation results will be created.

    Returns:
    --------
    results: dict
        Dictionary containing the segmentation results, with keys:
        - 'class_names': list of class names
        - 'scores': list of scores for each class (if available)
        - 'masks': list of binary masks for each class

    """
    results = {'class_names': [], 'scores': [], 'masks': []}
    for shape in data['shapes']:
        results['class_names'].append(shape['label'])
        if 'scores' in shape.keys():
            results['scores'].append(shape['scores']['cls'])
        else:
            results['scores'].append(None)

        mask_points = np.array(shape['points']).astype(np.int64)
        if shape['shape_type'] == 'rectangle':
            mask = np.zeros(shape_).astype('uint8')
            mask[mask_points[0,1]:mask_points[1,1], mask_points[0,0]: mask_points[1,0]] = 1
        else:
            mask = poly2mask(mask_points[:, 1], mask_points[:, 0], np.array(shape_)).astype('uint8')

        results['masks'].append(mask)

    for k, v in results.items():
        results[k] = np.array(v)

    return results
