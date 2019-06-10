import numpy as np
import functions as func
import matplotlib.pyplot as plt
import scipy.io
import os
import numpy.matlib
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


'''cube generation,'''

center=np.array([500,500,500]) # define center of the cube
length,width,height =(200,80,300) # define x_length,y_width,z_height

object = func.generate_cube(center,length,width,height)


'''define xray system, assume distance between xray and intensifier is 1000mm, and pixels on intensifier is 1000*1000, 
adjust by real parameter'''
source_point = np.array([1000,500,500])
screen_normal = np.array([1,0,0])
screen_point = np.array([0,500,500])

rotation = (0, 0, 0) # ratation angle around chosen axis
translation = (0, 0, 0) # direction and length to translate

proj_points,object = func.xray_project(object, source_point, screen_normal,screen_point, rotation, translation)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d',adjustable='box')
ax.scatter(object[:,0], object[:,1], object[:,2],c='k', marker='o')

faces_object=func.cube_faces(object)
faces_object.set_facecolor((0,0,1,0.1))
ax.add_collection3d(faces_object)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


ax.scatter(source_point[0],source_point[1],source_point[2],c='r', marker='*')
ax.text(source_point[0],source_point[1],source_point[2],'X-ray 1')
ax.scatter(proj_points[:,0], proj_points[:,1], proj_points[:,2],c='b', marker='o')

faces_proj1=func.cube_faces(proj_points)
faces_proj1.set_facecolor((0,0,1,0.1))
ax.add_collection3d(faces_proj1)

func.set_axes_equal(ax)
ax.view_init(30, 51)
plt.show()











