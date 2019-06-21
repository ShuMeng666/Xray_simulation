from __future__ import print_function
import numpy as np
import math
from math import cos, sin, radians
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy.matlib
import matplotlib.pyplot as plt


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


# transformation method 1
def trig(angle):
    r = radians(angle)
    return cos(r), sin(r)


def matrix(rotation, translation):
    xC, xS = trig(rotation[0])
    yC, yS = trig(rotation[1])
    zC, zS = trig(rotation[2])
    dX = translation[0]
    dY = translation[1]
    dZ = translation[2]
    Translate_matrix = np.array([[1, 0, 0, dX],
                                 [0, 1, 0, dY],
                                 [0, 0, 1, dZ],
                                 [0, 0, 0, 1]])
    Rotate_X_matrix = np.array([[1, 0, 0, 0],
                                [0, xC, -xS, 0],
                                [0, xS, xC, 0],
                                [0, 0, 0, 1]])
    Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                                [0, 1, 0, 0],
                                [-yS, 0, yC, 0],
                                [0, 0, 0, 1]])
    Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                                [zS, zC, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    return np.dot(Rotate_Z_matrix, np.dot(Rotate_Y_matrix, np.dot(Rotate_X_matrix, Translate_matrix)))


def transform(point, vector):
    p = [0, 0, 0]
    for r in range(3):
        p[r] += vector[r][3]
        for c in range(3):
            p[r] += point[c] * vector[r][c]
    return p


# transformation method 2
def rodrigues_rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    theta = (theta / 180) * 3.14;
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def generate_cube(center, x, y, z):
    p0 = np.full_like(center, 0)

    p0[0] = center[0] - x / 2
    p0[1] = center[1] - y / 2
    p0[2] = center[2] - z / 2

    p1 = p0.copy()
    p1[0] = p0[0] + x

    p2 = p0.copy()
    p2[1] = p0[1] + y

    p3 = p0.copy()
    p3[2] = p0[2] + z

    cube_definition_array = [p0, p1, p2, p3]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]
    points = np.array(points)
    return points


def cube_faces(points):
    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='none', alpha=0.2)
    return faces


def xray_project(object, source_point, screen_normal, screen_point, rotation, translation):
    matrix_temp = matrix(rotation, translation)
    rotated = np.full_like(object, 0)

    for i in range(0, len(object)):
        temp_p = transform(object[i, :], matrix_temp)
        rotated[i, 0] = temp_p[0]
        rotated[i, 1] = temp_p[1]
        rotated[i, 2] = temp_p[2]
    center_ori = np.transpose(object).mean(axis=1)
    center_trans = np.transpose(rotated).mean(axis=1)
    translate_value = numpy.matlib.repmat(center_trans - center_ori - translation, len(object), 1)
    object = rotated - translate_value

    proj_points_scn1 = np.full_like(object, 0)
    for i in range(0, len(object)):
        temp_ray_dirct_source1 = object[i, :] - source_point
        proj_points_scn1[i, :] = LinePlaneCollision(screen_normal, screen_point, temp_ray_dirct_source1, source_point)

    faces = cube_faces(proj_points_scn1)
    faces.set_facecolor((0, 0, 1, 0.1))

    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(faces)

    ax.view_init(azim=0, elev=0)
    ax.set_zlim(0, 1000)
    ax.set_ylim(0, 1000)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    fig.savefig('Proj_rotation' + str(rotation) + '_trans_' + str(translation) + '.jpg')

    return proj_points_scn1, object







