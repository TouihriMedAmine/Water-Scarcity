import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    """
    A ConvLSTM cell (single layer).
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(4 * hidden_channels)

    def forward(self, x, h, c):
        """
        Forward pass through the ConvLSTM cell.
        
        Parameters:
        x (Tensor): Input tensor at the current time step (batch_size, input_channels, height, width)
        h (Tensor): Previous hidden state (batch_size, hidden_channels, height, width)
        c (Tensor): Previous cell state (batch_size, hidden_channels, height, width)

        Returns:
        h (Tensor): New hidden state (batch_size, hidden_channels, height, width)
        c (Tensor): New cell state (batch_size, hidden_channels, height, width)
        """
        combined = torch.cat([x, h], dim=1)  # Concatenate along channel axis
        conv_output = self.conv(combined)
        conv_output = self.batch_norm(conv_output)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    """
    ConvLSTM model (multi-layer).
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # Define multiple ConvLSTM layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(input_channels if i == 0 else hidden_channels, hidden_channels, kernel_size)
            for i in range(num_layers)
        ])

        # Final convolutional layer to map hidden_channels to input_channels
        self.final_conv = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, channels, height, width)
        Returns:
            Output tensor of shape (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.size()
        h, c = [None] * self.num_layers, [None] * self.num_layers

        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(x_t, h[i] if h[i] is not None else torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device),
                                  c[i] if c[i] is not None else torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device))
                x_t = h[i]

        # Apply the final convolutional layer to the output of the last ConvLSTM cell
        output = torch.tanh(self.final_conv(h[-1]))  # Shape: (batch_size, input_channels, height, width)
        return output
