# Deep-learned channel estimator: 1D residual CNN in the delay domain.
#
# TensorFlow/Sionna port of the PUSCH DMRS channel-estimation CNN from NVIDIA
# pyAerial:
#   aerial-cuda-accelerated-ran/pyaerial/notebooks/channel_estimation
#   (channel_est_models.py: ChannelEstimator / ResidualBlock / ComplexMSELoss)
#
# Architecture (per receive antenna, over the subcarrier axis):
#   optional FFT to the delay domain  ->  Conv1D(2->C)+ReLU
#   -> ResidualBlock(dilation=1) -> ResidualBlock(dilation=3)
#   -> Conv1D(C->2)  ->  optional inverse FFT back to frequency.
# Each ResidualBlock is Conv1D-LayerNorm-ReLU-Conv1D-LayerNorm-(+skip)-ReLU.
#
# Integrated exactly like ARAChannelEstimator: the classical LS despreading +
# linear interpolation (Sionna PUSCHLSChannelEstimator) provides a full-grid
# estimate, and this CNN denoises it. The read-out conv is zero-initialised so
# an untrained CNN reproduces the LS+linear estimate (identity refinement),
# which keeps the LS / CNN / ARA comparison fair and prevents degradation
# below the LS baseline before training.

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

from sionna.nr import PUSCHLSChannelEstimator


# ---------------------------------------------------------------------------
# delay-domain transform (fftshift(fft(.)) / ifft(ifftshift(.)) over subcarriers)
# ---------------------------------------------------------------------------

def _sfft(x):
    """x: complex [N, F] -> fftshift(fft(x)) along the last axis."""
    return tf.signal.fftshift(tf.signal.fft(x), axes=-1)


def _isfft(x):
    """inverse of _sfft."""
    return tf.signal.ifft(tf.signal.ifftshift(x, axes=-1))


def _to_ri(xc):
    """complex [N, F] -> real [N, F, 2]."""
    return tf.stack([tf.math.real(xc), tf.math.imag(xc)], axis=-1)


def _to_complex(xr):
    """real [N, F, 2] -> complex [N, F]."""
    return tf.complex(xr[..., 0], xr[..., 1])


# ---------------------------------------------------------------------------
# Building blocks (pyAerial channel_est_models.py)
# ---------------------------------------------------------------------------

class CNNResidualBlock(Layer):
    """Conv1D-LN-ReLU-Conv1D-LN-(+skip)-ReLU with dilated 3-tap convolutions."""

    def __init__(self, channels, dilation, **kwargs):
        super().__init__(**kwargs)
        self._conv1 = tf.keras.layers.Conv1D(channels, 3, padding="same",
                                             dilation_rate=dilation, use_bias=False)
        self._norm1 = tf.keras.layers.LayerNormalization(axis=-1)
        self._conv2 = tf.keras.layers.Conv1D(channels, 3, padding="same",
                                             dilation_rate=dilation, use_bias=False)
        self._norm2 = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x):
        z = tf.nn.relu(self._norm1(self._conv1(x)))
        z = self._norm2(self._conv2(z))
        return tf.nn.relu(z + x)


class CNNChannelEstimatorNet(Model):
    """1D residual CNN denoiser over the subcarrier axis (delay domain).

    Input / output : [N, num_subcarriers, 2] (real/imag stacked), where N packs
    all (batch, rx, rx_ant, tx, stream, ofdm_symbol) into the batch dimension.
    Predicts a residual added to the (frequency-domain) input estimate; the
    read-out conv is zero-initialised => identity at initialisation.
    """

    def __init__(self, num_conv_channels=32, do_fft=True, **kwargs):
        super().__init__(**kwargs)
        self._do_fft = do_fft
        self._input_conv = tf.keras.layers.Conv1D(
            num_conv_channels, 3, padding="same", use_bias=False, activation="relu")
        self._res1 = CNNResidualBlock(num_conv_channels, dilation=1)
        self._res2 = CNNResidualBlock(num_conv_channels, dilation=3)
        # zero-init read-out => residual is zero at init (identity)
        self._output_conv = tf.keras.layers.Conv1D(
            2, 3, padding="same", use_bias=False, kernel_initializer="zeros")

    def call(self, z):
        # z: [N, F, 2] in the frequency domain
        zc = _to_complex(z)                       # [N, F] complex (freq)
        proc_in = _to_ri(_sfft(zc)) if self._do_fft else z

        h = self._input_conv(proc_in)
        h = self._res1(h)
        h = self._res2(h)
        out = self._output_conv(h)                # [N, F, 2] residual (delay if fft)

        oc = _to_complex(out)
        if self._do_fft:
            oc = _isfft(oc)                       # back to frequency domain
        return _to_ri(zc + oc)                    # residual add in frequency domain


# ---------------------------------------------------------------------------
# subcarrier-vector layout helper (shared by estimator and training)
# ---------------------------------------------------------------------------

def channel_to_subcarrier(h):
    """Sionna channel tensor -> per-antenna subcarrier vectors for the CNN.

    ``h`` : [B, num_rx, num_rx_ant, num_tx, num_streams(or num_tx_ant),
             num_ofdm_symbols, num_subcarriers], complex
    returns [N, num_subcarriers, 2] float32 (N packs all other dims).
    """
    F = tf.shape(h)[6]
    flat = tf.reshape(h, [-1, F])
    return tf.stack([tf.math.real(flat), tf.math.imag(flat)], axis=-1)


# ---------------------------------------------------------------------------
# Sionna-compatible channel estimator
# ---------------------------------------------------------------------------

class CNNChannelEstimator(PUSCHLSChannelEstimator):
    r"""LS + linear-interpolation estimate refined by a 1D delay-domain CNN.

    Drop-in replacement for ``PUSCHLSChannelEstimator`` inside a Sionna
    ``PUSCHReceiver`` (systems ``baseline_cnn_lmmse`` / ``baseline_cnn_kbest``).
    The CNN denoises each receive antenna's subcarrier vector independently,
    mirroring the pyAerial per-antenna channel estimator.

    Output shapes match ``PUSCHLSChannelEstimator``.
    """

    def __init__(self, resource_grid, dmrs_length, dmrs_additional_position,
                 num_cdm_groups_without_data,
                 num_conv_channels=32, do_fft=True, **kwargs):
        super().__init__(resource_grid=resource_grid,
                         dmrs_length=dmrs_length,
                         dmrs_additional_position=dmrs_additional_position,
                         num_cdm_groups_without_data=num_cdm_groups_without_data,
                         interpolation_type="lin",
                         dtype=tf.complex64,
                         **kwargs)
        self._cnn = CNNChannelEstimatorNet(num_conv_channels=num_conv_channels,
                                           do_fft=do_fft)

    def _denoise(self, h_hat, training=False):
        # h_hat: [B, R, A, T, S, O, F]; subcarriers F are already the last axis
        shp = tf.shape(h_hat)
        x = channel_to_subcarrier(h_hat)                     # [N, F, 2]
        y = self._cnn(x, training=training)                  # [N, F, 2]
        out = tf.complex(y[..., 0], y[..., 1])               # [N, F]
        return tf.reshape(out, shp)

    def call(self, inputs, training=False):
        h_hat, err_var = super().call(inputs)
        h_hat = self._denoise(h_hat, training=training)
        return h_hat, err_var
