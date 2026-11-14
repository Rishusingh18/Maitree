"""
The based unit of graph convolutional networks., based on awesome previous work by https://github.com/yysijie/st-gcn
"""

import torch
import torch.nn as nn
from metro_safety.anomaly.stg_nf.modules_pose import InvertibleConv1x1


class InvConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 LU_decomposed,
                 kernel_size):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)

    def forward(self, x, A, logdetA, logdet=None, reverse=False):
        n, kc, t, v = x.size()
        channels_per_part = kc // self.kernel_size
        graph_det = channels_per_part * t * logdetA[1]

        if not reverse:
            x_conv, conv_det = self.conv(x, logdet=None, reverse=False)
            x_view = x_conv.view(n, self.kernel_size, channels_per_part, t, v)
            x_test = x_view.permute(0, 2, 3, 1, 4).reshape(n, channels_per_part, t, -1)
            a_test = A.reshape(-1, v)
            x_out = torch.cat((x_view[:, 0], x_test @ a_test), dim=1).reshape(n, kc, t, v)

            step_det = conv_det + graph_det
            if logdet is not None:
                return x_out, logdet + step_det
            return x_out, step_det
        else:
            a1_inverse = A[1].inverse()
            x_view = x.view(n, self.kernel_size, channels_per_part, t, v)
            x0 = x_view[:, 0]
            x1 = (x_view[:, 1] - (x0 @ A[0])) @ a1_inverse
            x_rev = torch.cat((x0, x1), dim=1).reshape(n, kc, t, v)
            z, conv_det = self.conv(x_rev, logdet=None, reverse=True)

            step_det = conv_det - graph_det
            if logdet is not None:
                return z, logdet + step_det
            return z, step_det
