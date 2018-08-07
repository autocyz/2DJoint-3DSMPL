import os
import numpy as np


from SMPL.smpl_np import SMPLModel

if __name__ == '__main__':
    smpl = SMPLModel('model/feman_lbs.pkl')
    pose_path = 'data/test/pose/'
    joint_path = 'data/test/joint/'
    np.random.seed(2463)
    sample_num = 10000
    for i in range(sample_num):
        name = 'pose_%05d' % i
        pose = (np.random.rand(*smpl.pose_shape) - 0.5)
        np.save(os.path.join(pose_path, name), pose)
        # with open(os.path.join(pose_path, name), 'w') as fp:
        #     for line in pose:
        #         for ele in line:
        #             fp.write('%f\n' % ele)
        # fp.close()

        name = 'joint_%05d' % i
        smpl.set_params(pose=pose)
        joints = smpl.get_changed_joints()
        np.save(os.path.join(joint_path, name), joints)
        # with open(os.path.join(joint_path,name), 'w') as fp:
        #     for line in joints:
        #         for ele in line:
        #             fp.write('%f\n' % ele)
        # fp.close()
        print('generator: %05d\n' % i)

        # smpl.save_to_obj('data/v_%05d.obj' % i)
        # smpl.save_joints_to_obj('data/j_%05d.obj' % i)


