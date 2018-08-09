# coding: utf-8

import numpy as np
import pickle
import os

class SMPLModel():
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            params = pickle.load(f,encoding='iso-8859-1')

            self.J_regressor = params['J_regressor']
            self.weights = params['weights']
            self.posedirs = params['posedirs']
            self.v_template = params['v_template']
            self.shapedirs = params['shapedirs']
            self.faces = params['f']
            self.kintree_table = params['kintree_table']

        id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J = None
        self.R = None

        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template # how beta affect body shapE
        self.J = self.J_regressor.dot(v_shaped) # joints location
        pose_cube = self.pose.reshape((-1, 1, 3)) # (24,1,3)
        self.R = self.rodrigues(pose_cube) # rotation matrix for each joint (24,3,3)
        I_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0), (self.R.shape[0]-1, 3, 3))
        # (23,3,3) 23个3*3的单位矩阵

        lrotmin = (self.R[1:] - I_cube).ravel()
        # 23*9 =207 变成一个向量

        v_posed = v_shaped + self.posedirs.dot(lrotmin) # how pose affect body shape in zero pose
        G = np.empty((self.kintree_table.shape[1], 4, 4)) # world transformation of each joint
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
                )
            )
        T = np.tensordot(self.weights, G, axes=[[1], [0]]) # transformation of each vertex
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])

    def rodrigues(self, r):
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True) #对每个节点的三个值求取二范数
        # (24,1,1)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny) # 防止出现分母为0的情况
        r_hat = r / theta # (24,1,3)
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0]) # shape =(24, 0)
        m = np.dstack([
            z_stick,             -r_hat[:, 0, 2],    r_hat[:, 0, 1],
            r_hat[:, 0, 2],     z_stick,            -r_hat[:, 0, 0],
            -r_hat[:, 0, 1],    r_hat[:, 0, 0],     z_stick]
        ).reshape([-1, 3, 3]) # 旋转矩阵，（24*3*3）
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        ) # (24,3,3) 24个单位矩阵
        A = np.transpose(r_hat, axes=[0, 2, 1])# (24,3,1)
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m # (24,3,3)
        return R

    def with_zeros(self, x):
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))


    def pack(self, x):
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def get_changed_joints(self):
        return self.J_regressor.dot(self.verts)

    def save_joints_to_obj(self, path):
        joints = self.get_changed_joints()
        with open(path, 'w') as fp:
            for j in joints:
                fp.write('v %f %f %f\n' % (j[0], j[1], j[2]))


if __name__ == '__main__':
    smpl = SMPLModel('model/feman_lbs.pkl')
    np.random.seed(9608)
    # pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
    # beta = (np.random.rand(*smpl.beta_shape) - 0.5) * 0.06
    # pose = np.zeros(smpl.pose_shape)
    # pose[4,2] = 0
    trans = np.zeros(smpl.trans_shape)
    smpl.set_params()
    smpl.save_to_obj('./smpl_np.obj')
    smpl.save_joints_to_obj('./joints.obj')


