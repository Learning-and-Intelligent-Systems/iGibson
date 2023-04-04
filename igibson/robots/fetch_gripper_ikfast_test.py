import pyikfast_fetch as ik
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt

no_solutions = 0


xy = np.empty((10000, 2))
c = np.zeros(10000)
tries = []

for j in range(10000):
    if j % 1000 == 0:
        print(f'{j}-th sample')
    gripper_pos = np.random.rand(3) * np.array([4, 4, 0]) - np.array([2, 2, -0.386])
    xy[j] = gripper_pos[:2]
    gripper_orn = [0.707, 0, -0.707, 0]
    # gripper_orn = [1, 0, 0, -1]

    # This should be constant and I can get directly by printing from the Behavior code
    # gripper_to_wrist = ((-0.1665, 0, 0), (0, 0, 0, 1))
    gripper_to_wrist = ((-0.3165, 0, 0), (0, 0, 0, 1))


    target_world_to_wrist = p.multiplyTransforms(gripper_pos,
                                                 gripper_orn,
                                                 gripper_to_wrist[0],
                                                 gripper_to_wrist[1])


    # This one should vary depending on the torso's lift
    base_link_to_world = ((0, 0, -0.3860), (0, 0, 0, 1))
    base_link_to_target = p.multiplyTransforms(base_link_to_world[0],
                                               base_link_to_world[1],
                                               target_world_to_wrist[0],
                                               target_world_to_wrist[1])
    rel_pos = base_link_to_target[0]
    rel_orn = p.getMatrixFromQuaternion(base_link_to_target[1])
    for i in range(200):
        pfree = np.random.rand() * 3.2112 - 1.6056 
        joint_pos_ik = ik.inverse(list(rel_pos), list(rel_orn), pfree)
        if len(joint_pos_ik) > 0:
            c[j] = 1
            tries.append(i)
            break

plt.scatter(xy[c==0,0], xy[c==0,1], color='r')
plt.scatter(xy[c==1,0], xy[c==1,1], color='b')
plt.show()

plt.figure()
plt.hist(tries, bins=int(1+np.ceil(np.log(len(tries)))))
plt.show()
print(f'No solutions for {10000 - c.sum()} samples')