import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class Attn(nn.Module):

    def __init__(self):
        super(Attn, self).__init__()
        self.lr = 0.0001
        self.batch_size = 16
        self.cuda_exist = torch.cuda.is_available()
        print('cuda exist', self.cuda_exist)

        self.conv1 = nn.Conv2d(4, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        self.num_heads = 8
        self.w1 = nn.ModuleList([nn.Linear(26, 256) for _ in range(self.num_heads)])
        self.w2 = nn.ModuleList([nn.Linear(256, 256) for _ in range(self.num_heads)])
        self.w3 = nn.ModuleList([nn.Linear(256, 1) for _ in range(self.num_heads)])

        self.f_fc1 = nn.Linear(26 * self.num_heads, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_fc3 = nn.Linear(256, 6)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # prepare coord tensor
        self.coord_tensor = torch.FloatTensor(self.batch_size, 36, 2)
        if self.cuda_exist:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((self.batch_size, 36, 2))
        for i in range(36):
            np_coord_tensor[:,i,:] = np.array(self.cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        print('two heads')


    def cvt_coord(self, i):
        return [(i/6-2.5)/2.5, (i%6-2.5)/2.5]


    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        ## x = (bsize x 24 x 6 x 6)
        ## bsize is 1 test, 64 train
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (bsize x 36 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor[:mb]], dim=2)
        # (bsize x 36 x 26)
        x_flat2 = x_flat.view(mb*d*d, 26)

        objs = []
        for i in range(self.num_heads):
            scores = self.w3[i](selu(self.w2[i](selu(self.w1[i](x_flat2))))) # bsize*36 x 1
            scores = scores.squeeze(1).view(mb, d * d) # bsize x 36

            probs = F.softmax(scores).unsqueeze(1) # bsize x 1 x 36
            obj = torch.bmm(probs, x_flat).squeeze(1) # bsize x 26

            objs.append(obj)
        concat = torch.cat(objs, dim=1)

        x_f = self.f_fc1(concat)
        x_f = selu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = selu(x_f)
        x_f = F.dropout(x_f)
        x_f = self.f_fc3(x_f)
        
        return F.log_softmax(x_f)

    
    def train_(self, input_img, label):
        # check that input_img is torch Variable and channel in first dimension
        # make sure img is float 0. to 1.
        # input_img     | N, H, W, C
        # label         | N
        input_img = input_img.transpose(0, 3, 1, 2) / 255.
        input_img = torch.FloatTensor(input_img)
        label = torch.LongTensor(label)
        if self.cuda_exist:
            input_img = input_img.cuda()
            label = label.cuda()
        input_img = Variable(input_img)
        label = Variable(label)

        self.optimizer.zero_grad()
        output = self(input_img)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy
        

    def action_(self, input_img):
        # input_img     | H, W, C
        input_img = np.expand_dims(input_img, 0)
        input_img = input_img.transpose(0, 3, 1, 2) / 255.
        input_img = torch.FloatTensor(input_img)
        if self.cuda_exist:
            input_img = input_img.cuda()
        input_img = Variable(input_img)

        output = self(input_img)
        pred = output.data.max(1)[1]
        # bsize x 1
        return pred[0][0]


    def save_model(self, counter):
        torch.save(self.state_dict(), 'model-torch/counter_{}.pth'.format(counter))
