import numpy as np
import functions as func
import matplotlib.pyplot as plt
import scipy.io
import os
import numpy.matlib
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


#clear = lambda: os.system('clear')

center=np.array([500,512,512]) # define center of the cube
length,width,height =(200,80,300) # define x_length,y_width,z_height


object = func.generate_cube(center,length,width,height)




'''define xray system'''
source_point = np.array([1000,512,512])
screen_normal = np.array([1,0,0])
screen_point = np.array([0,512,512])


rotation = (0, 60, 0) # ratation angle around chosen axis
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


# '''if we need to rotate or translate the object, change the value to 1'''
# NeedToRotate = 1
#
#
# if NeedToRotate == 1:
#     # transformation method 1
#     rotation = (0, 0, 0) # ratation angle around chosen axis
#     translation = (0, 0, 0) # direction and length to translate
#     matrix = func.matrix(rotation, translation)
#
#     rotated = np.full_like(object,0)
#
#     for i in range(0,len(object)):
#         temp_p= func.transform(object[i,:],matrix)
#         rotated[i,0]=temp_p[0]
#         rotated[i,1]=temp_p[1]
#         rotated[i,2]=temp_p[2]
#     center_ori=np.transpose(object).mean(axis=1)
#     center_trans=np.transpose(rotated).mean(axis=1)
#
#     translate_value=numpy.matlib.repmat(center_trans-center_ori-translation,len(object),1)
#     object=rotated-translate_value
#
#

# transformation method 2, Euler–Rodrigues rotation

# axis = [1,0,0] #choose rotation axis
# theta = 90 #choose angle
# translation = np.array([10, 0, 0]) #direction and length to translate
# rotated = np.full_like(object,0)
#
# for i in range(0,len(object)):
#    temp_p=np.dot(func.rodrigues_rotation_matrix(axis, theta), object[i,:])
#    rotated[i,0]=temp_p[0]
#    rotated[i,1]=temp_p[1]
#    rotated[i,2]=temp_p[2]
#
#
# center_ori=np.transpose(object).mean(axis=1)
# center_trans=np.transpose(rotated).mean(axis=1)
#
# translate_value=numpy.matlib.repmat(center_trans-center_ori-translation,len(object),1)
# object=rotated-translate_value


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d',adjustable='box')
#
# ax.scatter(object[:,0], object[:,1], object[:,2],c='k', marker='o')
# faces_object=func.cube_faces(object)
# faces_object.set_facecolor((0,0,1,0.1))
# ax.add_collection3d(faces_object)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
#
# '''locations of X-Ray source points.we can adjust the locations depend on real distance between source and screen,
# '''
# source_p1= np.array([100,52,-20])
# ax.scatter(source_p1[0],source_p1[1],source_p1[2],c='r', marker='*')
# ax.text(source_p1[0],source_p1[1],source_p1[2],'X-ray 1')
#
#
# '''use 1 point and the plane normal to determine the screen (intensifier) plane, here for convenience we just use the
# plane x==0 for screen1 and y=0 for screen2
# '''
# screen1_p = np.array([0,1,1])
# screen1_normal = np.array([1,0,0])
#
# proj_points_scn1 = np.full_like(object,0)
#
#
# '''project the object to corresponding screens'''
# for i in range(0,len(object)):
#     temp_ray_dirct_source1 = object[i,:]-source_p1
#     proj_points_scn1[i,:] = func.LinePlaneCollision(screen1_normal, screen1_p, temp_ray_dirct_source1, source_p1)
#
#
# ax.scatter(proj_points_scn1[:,0], proj_points_scn1[:,1], proj_points_scn1[:,2],c='b', marker='o')
#
# faces_proj1=func.cube_faces(proj_points_scn1)
# faces_proj1.set_facecolor((0,0,1,0.1))
# ax.add_collection3d(faces_proj1)
#
#


# func.set_axes_equal(ax)
# ax.view_init(30, 51)
# plt.show()









