import torch
predicted = torch.zeros(2, 1, 17, 3)
target = torch.ones(2, 1, 17, 3)
dist_joints = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 3)
print(dist_joints.shape)
