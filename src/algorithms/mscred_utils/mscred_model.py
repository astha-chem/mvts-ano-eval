import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from src.algorithms.mscred_utils.convolution_lstm import ConvLSTM
from src.algorithms.algorithm_utils import PyTorchUtils

"""MSCRED adapted from https://github.com/SKvtun/MSCRED-Pytorch"""

class MSCREDModule(nn.Module, PyTorchUtils):
    def __init__(self, num_timesteps, attention, seed:int, gpu:int):
        super(MSCREDModule, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)  

        self.Conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1))
        self.ConvLSTM1 = ConvLSTM(in_channels=32, h_channels=[32], kernel_size=3, seq_len=num_timesteps,
                                  attention=attention)
        self.Conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                               padding=(0, 1, 1))
        self.ConvLSTM2 = ConvLSTM(in_channels=64, h_channels=[64], kernel_size=3, seq_len=num_timesteps,
                                  attention=attention)
        self.Conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 2, 2), stride=(1, 2, 2),
                               padding=(0, 1, 1))
        self.ConvLSTM3 = ConvLSTM(in_channels=128, h_channels=[128], kernel_size=3, seq_len=num_timesteps,
                                  attention=attention)
        self.Conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.ConvLSTM4 = ConvLSTM(in_channels=256, h_channels=[256], kernel_size=3, seq_len=num_timesteps,
                                  attention=attention)
        self.Deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.Deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2, padding=1)
        self.Deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.Deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        """
        input X with shape: (batch, num_channels, seq_len, height, width)
        """
        x_c1_seq = F.selu(self.Conv1(x)) 
        _, (x_c1, _) = self.ConvLSTM1(x_c1_seq)

        x_c2_seq = F.selu(self.Conv2(x_c1_seq))
        _, (x_c2, _) = self.ConvLSTM2(x_c2_seq)

        x_c3_seq = F.selu(self.Conv3(x_c2_seq))
        _, (x_c3, _) = self.ConvLSTM3(x_c3_seq)

        x_c4_seq = F.selu(self.Conv4(x_c3_seq))
        _, (x_c4, _) = self.ConvLSTM4(x_c4_seq)

        x_d4 = F.selu(self.Deconv4.forward(x_c4, output_size=[x_c3.shape[-1], x_c3.shape[-2]]))

        x_d3 = torch.cat((x_d4, x_c3), dim=1)
        x_d3 = F.selu(self.Deconv3.forward(x_d3, output_size=[x_c2.shape[-1], x_c2.shape[-2]]))

        x_d2 = torch.cat((x_d3, x_c2), dim=1)
        x_d2 = F.selu(self.Deconv2.forward(x_d2, output_size=[x_c1.shape[-1], x_c1.shape[-2]]))

        x_d1 = torch.cat((x_d2, x_c1), dim=1)
        x_rec = F.selu(self.Deconv1.forward(x_d1, output_size=[x_c1.shape[-1], x_c1.shape[-2]]))

        # X_rec - reconstructed signature matrix at last time step

        return x_rec