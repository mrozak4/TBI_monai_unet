from cc3d import connected_components
from numpy import unique, delete, zeros, isin, hstack, asarray, sum, min, argmin
from vg import normalize, perpendicular, angle, basis
from scipy.spatial.transform import Rotation
from pytransform3d.rotations import matrix_from_axis_angle
import scipy as sp

def remove_small_comps_3d(image, thresh=500):
    """
    Remove small connected components from a 3D image.

    Parameters:
        image (ndarray): The input 3D image.
        thresh (int): The threshold value to determine the minimum size of connected components to keep.

    Returns:
        ndarray: The filtered image with small connected components removed.
    """
    # Convert the input image to a labeled image where each connected component is assigned a unique label
    img_lab, N = connected_components(image, return_N=True)

    # Count the number of pixels for each unique label
    _unique, counts = unique(img_lab, return_counts=True)

    # Keep only the labels that have pixel count greater than the threshold
    unique_keep = _unique[counts > thresh]

    # Remove the background label (label 0)
    unique_keep = delete(unique_keep, [0])

    # Create a filtered image where only the connected components with labels in unique_keep are kept
    img_filt = zeros(img_lab.shape).astype('int8')
    img_filt[isin(img_lab, unique_keep)] = 1

    return img_filt.astype('uint8')

def fill_holes(img, thresh=1000):
    """
    Fills holes in a binary image using connected component analysis.

    Args:
        img (ndarray): The input binary image.
        thresh (int): The threshold value for removing small components. Default is 1000.

    Returns:
        ndarray: The binary image with filled holes.

    """
    # Iterate over unique values in the image in reverse order
    for i in unique(img)[::-1]:
        # Create a temporary binary image with only the current value
        _tmp = (img == i) * 1.0
        _tmp = _tmp.astype('int8')
        
        # Remove small components from the temporary image
        _tmp = remove_small_comps_3d(_tmp, thresh=thresh)
        
        # Update the original image with the filled holes
        img[_tmp == 1] = i
    
    # Convert the image to int8 type and return
    res = img.astype('int8')
    return res

def _rotmat(vector, points):
    """
    Rotate the given points around the given vector.

    Args:
        vector (array-like): The vector around which the points should be rotated.
        points (array-like): The points to be rotated.

    Returns:
        array-like: The rotated points.
    """
    # Normalize the vector
    vector = normalize(vector)

    # Calculate the axis of rotation
    axis = perpendicular(basis.z, vector)

    # Calculate the angle of rotation in radians
    _angle = angle(basis.z, vector, units='rad')

    # Create the rotation matrix
    a = hstack((axis, (_angle,)))
    R = matrix_from_axis_angle(a)

    # Apply the rotation matrix to the points
    r = Rotation.from_matrix(R)
    rotmat = r.apply(points)

    return rotmat

def closest_node(node, nodes, maximal_distance=10):
    """
    Find the closest node to a given node from a list of nodes.

    Parameters:
        node (array-like): The node for which the closest node needs to be found.
        nodes (array-like): The list of nodes to search from.
        maximal_distance (float): The maximum distance allowed for a node to be considered as the closest. Default is 10.

    Returns:
        array-like: The closest node to the given node from the list of nodes.
    """
    # Convert nodes to a numpy array for efficient computation
    nodes = asarray(nodes)

    # Calculate the squared Euclidean distance between each node and the given node
    dist_2 = sum((nodes - node)**2, axis=1)

    # Check if the minimum distance is greater than 10
    if min(dist_2) > maximal_distance:
        # If the minimum distance is greater than 10, return the given node itself
        return node
    else:
        # If the minimum distance is less than or equal to 10, return the node with the minimum distance
        return nodes[argmin(dist_2)]