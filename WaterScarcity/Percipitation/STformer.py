import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from torch import einsum
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class ConvLSTM2DCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Convolutional gates
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # for input, forget, cell, output gates
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Combine input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Compute all gates
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply nonlinearities
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Update cell state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvLSTM2DCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                bias=bias
            ))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, _, height, width = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.batch_first:
            layer_output = layer_output.permute(1, 0, 2, 3, 4)

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class VisionTransformer(nn.Module):
    def __init__(self, input_len=8, target_len=8, image_size=(64,64), dim=256, num_heads=4, num_layers=6):
        super().__init__()
        self.input_len = input_len
        self.target_len = target_len
        self.image_size = image_size
        self.dim = dim
        H, W = image_size
        
        self.patch_size = 16
        self.num_patches = (H // self.patch_size) * (W // self.patch_size)
        self.patch_dim = 1 * self.patch_size * self.patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Positional encodings
        self.temporal_pos = nn.Parameter(0.02 * torch.randn(1, input_len, dim))
        self.spatial_pos = nn.Parameter(0.02 * torch.randn(1, self.num_patches, dim))
        self.dynamic_pos = nn.Parameter(torch.randn(1, target_len, dim))

        self.dropout = nn.Dropout(0.3)
        
        # Temporal processing with ConvLSTM
        self.conv_lstm = ConvLSTM2D(
            input_dim=dim,
            hidden_dim=dim,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(dim)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        
        # Skip connection
        self.skip_conv = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, self.patch_dim)
        )

    def forward(self, x):
        B, T, _, H, W = x.shape
        
        # === Patch Embedding ===
        x_patches = rearrange(x, 'b t c h w -> (b t) c h w')
        x_patches = self.patch_embed(x_patches)  # (B*T, D, H', W')
        skip = self.skip_conv(x_patches)
        
        # Get ConvLSTM input dimensions
        lstm_h = H // self.patch_size
        lstm_w = W // self.patch_size
        
        # ConvLSTM processing
        lstm_in = rearrange(x_patches, '(b t) c h w -> b t c h w', b=B, t=T)
        lstm_out, _ = self.conv_lstm(lstm_in)
        lstm_out = lstm_out[0]  # Take output from first (only) layer
        lstm_out = rearrange(lstm_out, 'b t c h w -> (b t) c h w')
        
        # Combine with original features
        x_patches = x_patches + lstm_out  # Residual connection
        x_patches = rearrange(x_patches, '(b t) d h w -> b t (h w) d', b=B, t=T)
        
        # === Temporal Processing ===
        x_temp = rearrange(x_patches, 'b t n d -> (b n) t d')
        x_temp = x_temp + self.dynamic_pos[:, :T]
        x_temp, _ = self.temporal_attention(x_temp, x_temp, x_temp)
        x_temp = self.temporal_norm(x_temp)
        x_temp = rearrange(x_temp, '(b n) t d -> b t n d', b=B, n=self.num_patches)
        
        # === Combine Features ===
        x = x_patches + x_temp  # Residual connection
        
        # Add positional encodings
        x = x + self.spatial_pos.unsqueeze(1)  # Add spatial pos
        x = x + self.temporal_pos.unsqueeze(2)  # Add temporal pos
        
        x = self.dropout(x)
        
        # === Transformer Processing ===
        x = rearrange(x, 'b t n d -> (b t) n d')  # Combine batch and temporal
        x = self.transformer(x)
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
        
        # === Prediction ===
        skip = rearrange(skip, '(b t) d h w -> b t (h w) d', b=B, t=T)
        future_patches = self.head(x + skip)  # Add skip connection
        
        # Reshape patches back to images
        future_patches = future_patches.view(B, T, self.num_patches, 1, self.patch_size, self.patch_size)
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        future_patches = future_patches.view(B, T, grid_h, grid_w, 1, self.patch_size, self.patch_size)
        future_images = future_patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        future_images = future_images.view(B, T, 1, H, W)
        
        # Post-processing
        threshold = 0.91
        white_mask = (future_images > threshold).all(dim=2, keepdim=True)
        future_images = future_images.masked_fill(white_mask, 1.0)
        
        return future_images