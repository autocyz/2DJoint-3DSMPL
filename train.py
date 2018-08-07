import numpy as np
import torch
import torch.nn as nn
from logger import Logger
import torchvision.transforms as transforms
import os
from model import MyModel


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, pose_path, joint_path, transform=None, transform_target=None):
        self.pose_path = pose_path
        self.joint_path = joint_path
        self.transform = transform
        self.transform_target = transform_target
        self.pose_list = []
        self.joint_list = []

        for file in os.listdir(pose_path):
            pose_file = os.path.join(self.pose_path, file)
            if os.path.isfile(pose_file):
                self.pose_list.append(pose_file)
        for file in os.listdir(joint_path):
            joint_file = os.path.join(self.joint_path, file)
            if os.path.isfile(joint_file):
                self.joint_list.append(joint_file)

        self.pose_list.sort()
        self.joint_list.sort()

    def __getitem__(self, index):
        pose = np.load(self.pose_list[index])
        joint = np.load(self.joint_list[index])
        joint = joint[:, 0:2]
        pose = pose.reshape(-1)
        # pose += 0.5
        if self.transform is not None:
            joint = self.transform(joint)
        if self.transform_target is not None:
            pose = self.transform_target(pose)
        return joint, pose

    def __len__(self):
        return len(self.pose_list)


def adjust_learning_rate(optimizer, epoch, decay_rate, decay_step):
    """Sets the learning rate to the initial LR decayed by decay_rate every decay_step epochs"""
    for param_group in optimizer.param_groups:
        if epoch != 0 and epoch % decay_step == 0:
            param_group['lr'] = param_group['lr']*decay_rate


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
input_size = 24*2
output_size = 24*3
num_epochs = 200
batch_size = 64
learning_rate = 1e-3

logger = Logger('./train_logs/2018_08_07')
train_set = MyDataset('data/train/pose', 'data/train/joint')
test_set = MyDataset('data/test/pose', 'data/test/joint')
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=True)

model = MyModel().to(device)
model.double()

# criterion = nn.MSELoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
print(total_step)
last_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    for i, (joint, pose) in enumerate(train_loader):
        joint = joint.to(device)
        pose = pose.to(device)

        # Forward pass
        output = model(joint)
        loss = criterion(output, pose)
        # loss *= 100
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], lr: {} Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, float(optimizer.param_groups[0]['lr']), loss.item()))

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss.item()}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch*total_step+i)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch*total_step+i)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch*total_step+i)

        # 3. Log training images (image summary)
        # info = {'images': joint.view(-1, 24, 2)[:10].cpu().numpy()}
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, step+1)

    loss = 0
    if epoch % 1 == 0:
        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            for joint, pose in test_loader:
                joint = joint.to(device)
                pose = pose.to(device)
                outputs = model(joint)
                loss += criterion(outputs, pose)
            loss = loss / len(test_loader)
            print('Test Loss of the model on the {} test images: {:.4f} '.format(total_step*batch_size, loss))
            logger.scalar_summary('test_loss', loss.item(), epoch)
    if loss < last_loss:
        last_loss = loss
        # Save the model checkpoint
        torch.save(model.state_dict(), 'trained_model/2018_08_06/epoch_%d_model.ckpt' % epoch)

