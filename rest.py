import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model_attn import *

net = Attn()
#net.cuda()

img = np.ones((84, 84, 4))
img = Variable(torch.FloatTensor(img).cuda())
img = img.permute(2, 0, 1).unsqueeze(0).repeat(64, 1, 1, 1)
out = net(img)

print(out)

