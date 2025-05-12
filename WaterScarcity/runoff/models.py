
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k):
        super().__init__()
        p = k // 2
        self.conv   = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=p)
        self.hid_ch = hid_ch

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates    = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        o = torch.sigmoid(o); g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        h = torch.zeros(batch_size, self.hid_ch, height, width, device=device)
        c = torch.zeros(batch_size, self.hid_ch, height, width, device=device)
        return h, c

class ConvLSTMForecaster(nn.Module):
    def __init__(self, in_ch, hid_ch, k, T):
        super().__init__()
        self.T        = T
        self.cell     = ConvLSTMCell(in_ch, hid_ch, k)
        self.conv_out = nn.Conv2d(hid_ch, 1, kernel_size=1)

    def forward(self, x):
        B, _, _, H, W = x.size()
        h, c = self.cell.init_hidden(B, H, W, x.device)
        for t in range(self.T):
            h, c = self.cell(x[:, t], h, c)
        return self.conv_out(h)
