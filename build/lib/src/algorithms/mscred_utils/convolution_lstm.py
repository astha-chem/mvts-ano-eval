'''
https://github.com/Zhang-Zhi-Jie/Pytorch-MSCRED, https://github.com/SKvtun/MSCRED-Pytorch
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, in_channels=32, h_channels=[32], kernel_size=3, seq_len=5, attention=True):

        super(ConvLSTM, self).__init__()
        self.input_channels = [in_channels] + h_channels
        self.hidden_channels = h_channels
        self.attention = attention
        self.kernel_size = kernel_size
        self.num_layers = len(h_channels)
        self.seq_len = seq_len
        self._all_layers = []
        self.flatten = nn.Flatten()
        self.alpha_i = None
        for i in range(self.num_layers):
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size).cuda()
            self._all_layers.append(cell)

    def forward(self, input):

        """
        input with shape: (batch, num_channels, seq_len, height, width)
        """
        internal_state = []
        outputs = []
        for timestep in range(self.seq_len):
            x = input[:, :, timestep, ...]
            for i in range(self.num_layers):
                if timestep == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = self._all_layers[i].init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                (h, c) = internal_state[i]
                x, new_c = self._all_layers[i](x, h, c)
                internal_state[i] = (x, new_c)

            outputs.append(x)
        outputs = torch.stack(outputs).permute(1, 2, 0, 3, 4)

        if self.attention:
            alpha_i = [torch.einsum('ij,ik->i', self.flatten(outputs[:, :, i, ...]), self.flatten(outputs[:, :, -1, ...]))
                       for i in range(self.seq_len)]
            alpha_i = torch.stack(alpha_i, dim=1) / self.seq_len
            alpha_i = F.softmax(alpha_i, dim=1)
            self.alpha_i = alpha_i
            x = torch.einsum('ijklm, ik -> ijlm', outputs, alpha_i)

        return outputs, (x, new_c)