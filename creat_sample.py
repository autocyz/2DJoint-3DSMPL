import os
import numpy as np


from SMPL.smpl_np import SMPLModel

if __name__ == '__main__':
    smpl = SMPLModel('model/feman_lbs.pkl')
    pose_path = '/mnt/data/dataset/SMPL/test/pose/'
    joint_path = '/mnt/data/dataset/SMPL/test/joint/'
    np.random.seed(3635)
    sample_num = 10000
    for i in range(sample_num):
        name = 'pose_%07d' % i
        pose = (np.random.rand(*smpl.pose_shape) - 0.5)
        np.save(os.path.join(pose_path, name), pose)
        # with open(os.path.join(pose_path, name), 'w') as fp:
        #     for line in pose:
        #         for ele in line:
        #             fp.write('%f\n' % ele)
        # fp.close()

        name = 'joint_%07d' % i
        smpl.set_params(pose=pose)
        joints = smpl.get_changed_joints()
        joints_14 = np.array((joints[15], (joints[16]+joints[17])/2, joints[17], joints[19], joints[21], joints[16],
                              joints[18], joints[20], joints[2], joints[5], joints[8], joints[1], joints[4],
                     joints[7]))


        # norm all the poins, select zero point and let all points
        # minus zero points and div the height.
        # Usually, height > width, so we select height to norm
        zero_point = (joints_14[8] + joints_14[11]) / 2
        joint_max = joints_14.max(0)
        joint_min = joints_14.min(0)
        joint_border = joint_max-joint_min
        joints_14 = (joints_14-zero_point)/joint_border[1]

        np.save(os.path.join(joint_path, name), joints_14)
        # with open(os.path.join(joint_path,name), 'w') as fp:
        #     for line in joints:
        #         for ele in line:
        #             fp.write('%f\n' % ele)
        # fp.close()
        print('generator: %07d\n' % i)

        # smpl.save_to_obj('data/v_%05d.obj' % i)
        # smpl.save_joints_to_obj('data/j_%05d.obj' % i)


