# Overview

This project is used to map 2D person joints to 3D SMPL model person joints. The input of SMPL model is pose and shape parameters. We want to generate joints by SMPL, and shape parameters don't affect joints' position, so we don't care shape parameters, just use pose parameters which size is 24*3.

We want to set up a mapping function, which can get input 2D joints  , and return SMPL pose parameters,  so that we can use these pose parameters to generate 3D joints. We use neural network to set up this mapping function.

### network 

We just use fully-connected layer. The model is similar with [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)  implemented by Pytorch.      

# File

We use SMPL model code from https://github.com/CalciferZh/SMPL , this code is much easier than official  [SMPL code](http://smpl.is.tue.mpg.de/).     

- `SMPL/smpl_np.py` use numpy to create SMPL mode
- `SMPL/smpl_tf.py` use tensorflow to create SMPL mode
- `creat_sample.py` create train set and test set for our network 
- `model.py` our network 
- `train.py` train network
- `test.py` test network

# Tips
use MeshLab display 3D model
```
sudo apt install meshlab
```
