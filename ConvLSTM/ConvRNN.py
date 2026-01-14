import torch
import torch.nn as nn

class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):

        super(CLSTM_cell, self).__init__()
        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2

        # No BatchNorm
        self.input_norm = nn.Identity()
        self.conv = nn.Conv2d(
                in_channels=self.input_channels + self.num_features,
                out_channels=4 * self.num_features,
                kernel_size=self.filter_size,
                stride=1,
                padding=self.padding,
                bias=True)

        #f = σ(1) ≈ 0.73。
        with torch.no_grad():
            C = self.num_features
            self.conv.bias[C:2*C].fill_(3.0)   # forget gate bias += 1

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                            self.shape[1], device=inputs.device)
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                            self.shape[1], device=inputs.device)
        else:
            hx, cx = hidden_state

        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0], self.shape[1],
                device=hx.device)
            else:
                x = inputs[index, ...]

            x = self.input_norm(x) 

            combined = torch.cat((x, hx), 1)

            gates = self.conv(combined)  
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)
