import numpy as np
import torch
import torch.nn as nn

from SMPL.smpl_np import SMPLModel
from model import MyModel
input_size = 14*2
output_size = 24*3
smpl = SMPLModel('model/feman_lbs.pkl')
model = MyModel(input_size,output_size )
model.double()
model.eval()
model.load_state_dict(torch.load('trained_model/2018_08_20/epoch_46_model.ckpt'))


joint_t = np.load('/mnt/data/dataset/SMPL/test/joint/joint_0000057.npy')
pose_t = np.load('/mnt/data/dataset/SMPL/test/pose/pose_0000057.npy')
joint_t = torch.from_numpy(joint_t[:, 0:2])
joint_t = joint_t.reshape(1, -1)
pose_p = model(joint_t).data.numpy()
pose_p = pose_p.reshape(24, 3)


smpl.set_params(pose=pose_t)
smpl.save_to_obj('test_result/vertices_t.obj')
smpl.save_joints_to_obj('test_result/joint_t.obj')

smpl.set_params(pose=pose_p)
smpl.save_to_obj('test_result/vertices_p.obj')
smpl.save_joints_to_obj('test_result/joint_p.obj')


# if np.allclose(joint_t, smpl.get_changed_joints()[]):
#     print('Bingo')

print(joint_t[0:3])
print(smpl.get_changed_joints()[0:3])
