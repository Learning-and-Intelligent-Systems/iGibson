from mesh_renderer_vr import MeshRendererVR
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
import cv2
import sys
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import VisualObject, InstanceGroup, MeshRenderer
import time

renderer = MeshRendererVR(MeshRenderer)
# Note that it is necessary to load the full path of an object!
#renderer.load_object("C:\\Users\\shen\\Desktop\\GibsonVRStuff\\gibsonv2\\gibson2\\assets\\models\\mjcf_primitives\\cube.obj")
renderer.load_object("C:\\Users\\shen\\Desktop\\GibsonVRStuff\\gibsonv2\\gibson2\\assets\\datasets\\Ohoopee\\Ohoopee_mesh_texture.obj")
renderer.add_instance(0)

camera_pose = np.array([0, 0, 1.2])
view_direction = np.array([1, 0, 0])
renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
renderer.set_fov(90)

renderer.setupDebugFramebuffer()

#camera_pose = np.array([0, 0, 1.2])
#renderer.set_vr_camera(camera_pose)

while True:
    renderer.renderDebugFramebuffer()
    #frame = renderer.render(vrMode=False)

    #cv2.imshow('VR Output (left eye - right eye)', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
    # Needed to actually display the image
    q = cv2.waitKey(50)

renderer.release()

# print(renderer.visual_objects, renderer.instances)
# print(renderer.materials_mapping, renderer.mesh_materials)
# camera_pose = np.array([0, 0, 1.2])
# view_direction = np.array([1, 0, 0])
# renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
# renderer.set_fov(90)

# while True:
#     frame = renderer.render(modes=('rgb'))
#     cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))

# renderer.release()