import torch
import torch.nn as nn
from seriesDecomp import SeriesDecomp
from villanseriesDecomp import villanSeriesDecomp
import torch.nn.functional as F
class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(
        self,
        attention,
        hidden_size,
        conv_hidden_size=None,
        MovingAvg=25,
        dropout=0.1,
        activation="relu",
        gruop_dec=True,
    ):
        super(EncoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=conv_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
            bias=False,
        )
        if gruop_dec:
            self.decomp1 = SeriesDecomp(MovingAvg,hidden_size)
            self.decomp2 = SeriesDecomp(MovingAvg,hidden_size)
        else:
            self.decomp1 = villanSeriesDecomp(MovingAvg, hidden_size)
            self.decomp2 = villanSeriesDecomp(MovingAvg, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, edge_index, attn_mask=None):

        all_aggregate = []
        all_attention = []
        for batch in range(x.shape[0]):

            each_batch_data = x[batch].unsqueeze(0)
            output, attn= self.attention(each_batch_data, each_batch_data, each_batch_data)
            each_batch_data = each_batch_data + self.dropout(output)

            all_aggregate.append(each_batch_data)
            all_attention.append(attn)

        x = torch.cat(all_aggregate, dim=0)
        # x = self.enc_embedding(x)
        x, _ = self.decomp1(x, edge_index)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y, edge_index)
        return res, all_attention

class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, edge_index, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, edge_index, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, edge_index, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns
