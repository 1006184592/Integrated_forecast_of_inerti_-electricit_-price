import torch
import torch.nn as nn
from Encoder import Encoder,EncoderLayer
from Decoder import Decoder,DecoderLayer
from DataEmbedding import GCNWithEmbeddings
from seriesDecomp import SeriesDecomp
from attention import AutoCorrelation,AutoCorrelationLayer
class LayerNorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

class Hier_auto(nn.Module):
    """Autoformer

    The Autoformer model tackles the challenge of finding reliable dependencies on intricate temporal patterns of long-horizon forecasting.

    The architecture has the following distinctive features:
    - In-built progressive decomposition in trend and seasonal compontents based on a moving average filter.
    - Auto-Correlation mechanism that discovers the period-based dependencies by
    calculating the autocorrelation and aggregating similar sub-series based on the periodicity.
    - Classic encoder-decoder proposed by Vaswani et al. (2017) with a multi-head attention mechanism.

    The Autoformer model utilizes a three-component approach to define its embedding:
    - It employs encoded autoregressive features obtained from a convolution network.
    - Absolute positional embeddings obtained from calendar features are utilized.

    *Parameters:*<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, maximum sequence length for truncated train backpropagation. Default -1 uses all history.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `exclude_insample_y`: bool=False, the model skips the autoregressive features y[t-input_size:t] if True.<br>
        `decoder_input_size_multiplier`: float = 0.5, .<br>
    `hidden_size`: int=128, units of embeddings and encoders.<br>
    `n_head`: int=4, controls number of multi-head's attention.<br>
    `dropout`: float (0, 1), dropout throughout Autoformer architecture.<br>
        `factor`: int=3, Probsparse attention factor.<br>
        `conv_hidden_size`: int=32, channels of the convolutional encoder.<br>
        `activation`: str=`GELU`, activation from ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid', 'GELU'].<br>
    `encoder_layers`: int=2, number of layers for the TCN encoder.<br>
    `decoder_layers`: int=1, number of layers for the MLP decoder.<br>
    `distil`: bool = True, wether the Autoformer decoder uses bottlenecks.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.<br>
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.<br>
    `inference_windows_batch_size`: int=1024, number of windows to sample in each inference batch.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `scaler_type`: str='robust', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.<br>
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

        *References*<br>
        - [Wu, Haixu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting"](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html)<br>
    """

    # Class attributes
    SAMPLING_TYPE = "windows"

    def __init__(
        self,
        n_head: int,
        hidden_size: int,
        factor : int= 2,
        dropout : float= 0.05,
        conv_hidden_size : int= 32,
        MovingAvg_window : int= 3,
        activation : str="gelu",
        encoder_layers : int= 2,
        decoder_layers : int= 1,
        c_out : int= 1,
        h: int = 1,
        seq_lenth: int = 6,
        c_in: int = 1,
        gruop_dec: bool=True

    ):
        super(Hier_auto, self).__init__()

        if activation not in ["relu", "gelu"]:
            raise Exception(f"Check activation={activation}")

        self.c_out = c_out
        self.output_attention = False

        # Decomposition
        self.enc_embedding = GCNWithEmbeddings(exog_input_size=0,hidden_size=hidden_size,
                                                c_in=c_in, seq_lenth=seq_lenth, dropout=dropout,)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=self.output_attention,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size=hidden_size,
                    conv_hidden_size=conv_hidden_size,
                    MovingAvg=MovingAvg_window,
                    dropout=dropout,
                    activation=activation,
                    gruop_dec=gruop_dec
                )
                for l in range(encoder_layers)
            ],
            norm_layer=LayerNorm(hidden_size),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size=hidden_size,
                    c_out=self.c_out,
                    dec_embedding=GCNWithEmbeddings(exog_input_size=0,hidden_size=hidden_size,
                                        c_in=c_in, seq_lenth=seq_lenth+c_out, dropout=dropout),
                    h=h,
                    c_in= c_in,
                    conv_hidden_size=conv_hidden_size,
                    MovingAvg=MovingAvg_window,
                    dropout=dropout,
                    activation=activation,
                    gruop_dec=gruop_dec
                )
                for l in range(decoder_layers)
            ],
            norm_layer=LayerNorm(hidden_size),
            projection=nn.Linear(hidden_size, h, bias=True),
        )

    def forward(self, windows_batch, edge_index):
        enc_windows_batch = self.enc_embedding(windows_batch, edge_index)
        enc_out, _ = self.encoder(enc_windows_batch, edge_index)
        dec_out = self.decoder(windows_batch, enc_out, edge_index)
        forecast = dec_out[:, -self.c_out:]
        return forecast
