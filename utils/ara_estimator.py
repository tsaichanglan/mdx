# Deep-learned channel estimator: Attentive Residual Autoencoder (ARA) network.
#
# Implements the channel-estimation network of
#   Y. Wei et al., "Deep-Learned Channel Estimation for MIMO-OFDM System by
#   Exploiting Frequency-Space Correlation", Electronics Letters, 2026.
#
# The ARA network replaces the interpolation/denoising stage of a classical
# 5G PUSCH channel estimator. Following the three-stage scheme of the paper
# (Fig. 2):
#   1) LS despreading of the pilots,
#   2) linear interpolation of the despread frequency-domain estimates,
#   3) ARA network denoising over the frequency-space plane.
# Stages 1-2 are inherited from Sionna's ``PUSCHLSChannelEstimator``
# (interpolation_type="lin"); stage 3 is the ARA network implemented here.
#
# Mahdi-style integration for the MDX repository.

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

from sionna.nr import PUSCHLSChannelEstimator


# ---------------------------------------------------------------------------
# Building blocks (paper Fig. 4)
# ---------------------------------------------------------------------------

def _groupnorm(channels):
    """GroupNorm that normalises every 4 channels as a group (paper, Sec. 1)."""
    groups = max(1, channels // 4)
    # ensure divisibility
    while channels % groups != 0:
        groups -= 1
    return tf.keras.layers.GroupNormalization(groups=groups, axis=-1)


class ResModule(Layer):
    """Res module (Fig. 4a): GN-ReLU-Conv -> GN-ReLU-Conv -> Dropout, residual."""

    def __init__(self, channels, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self._channels = channels
        self._gn1 = _groupnorm(channels)
        self._conv1 = tf.keras.layers.Conv2D(channels, 3, padding="same")
        self._gn2 = _groupnorm(channels)
        self._conv2 = tf.keras.layers.Conv2D(channels, 3, padding="same")
        self._dropout = tf.keras.layers.Dropout(dropout_rate)
        self._proj = None

    def build(self, input_shape):
        # 1x1 projection on the skip path when channel count changes
        if input_shape[-1] != self._channels:
            self._proj = tf.keras.layers.Conv2D(self._channels, 1, padding="same")
        super().build(input_shape)

    def call(self, x, training=False):
        shortcut = x if self._proj is None else self._proj(x)
        h = self._conv1(tf.nn.relu(self._gn1(x)))
        h = self._conv2(tf.nn.relu(self._gn2(h)))
        h = self._dropout(h, training=training)
        return shortcut + h


def _fir_blur_kernel(channels):
    """2D FIR (binomial) anti-alias kernel used before down/after up sampling."""
    k1 = np.array([1., 2., 1.], dtype=np.float32)
    k2 = np.outer(k1, k1)
    k2 = k2 / k2.sum()
    # depthwise kernel: [h, w, channels, 1]
    k = np.tile(k2[:, :, None, None], (1, 1, channels, 1))
    return tf.constant(k)


class FIRBlur(Layer):
    """Depthwise 2D FIR smoothing filter to mitigate (up/down)sampling aliasing."""

    def build(self, input_shape):
        self._k = _fir_blur_kernel(int(input_shape[-1]))
        super().build(input_shape)

    def call(self, x):
        return tf.nn.depthwise_conv2d(x, self._k, strides=[1, 1, 1, 1],
                                      padding="SAME")


class ResDown(Layer):
    """Res-down module (Fig. 4b): FIR smoothing + strided-conv downsampling."""

    def __init__(self, channels, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self._res = ResModule(channels, dropout_rate)
        self._fir = FIRBlur()
        self._down = tf.keras.layers.Conv2D(channels, 3, strides=2,
                                            padding="same")

    def call(self, x, training=False):
        x = self._res(x, training=training)
        x = self._fir(x)
        return self._down(x)


class ResUp(Layer):
    """Res-up module (Fig. 4c): interpolation upsampling + FIR + conv."""

    def __init__(self, channels, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self._res = ResModule(channels, dropout_rate)
        self._up = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self._fir = FIRBlur()
        self._conv = tf.keras.layers.Conv2D(channels, 3, padding="same")

    def call(self, x, target_hw=None, training=False):
        x = self._res(x, training=training)
        x = self._up(x)
        x = self._fir(x)
        x = self._conv(x)
        if target_hw is not None:
            x = x[:, :target_hw[0], :target_hw[1], :]
        return x


class SpatialAttention(Layer):
    """2D frequency-space (self- or cross-) attention (paper Eqs. 5-7).

    Queries are generated from ``x``; keys/values from ``context`` (``context``
    defaults to ``x`` for self-attention). Attention is computed over the full
    H x W frequency-space plane. To keep the cost feasible for very wide
    bandwidths, attention is skipped (identity) when H*W exceeds ``max_tokens``.
    """

    def __init__(self, channels, max_tokens=8192, **kwargs):
        super().__init__(**kwargs)
        self._channels = channels
        self._max_tokens = max_tokens
        self._gn_q = _groupnorm(channels)
        self._gn_k = _groupnorm(channels)
        self._gn_v = _groupnorm(channels)
        self._q = tf.keras.layers.Conv2D(channels, 1)
        self._k = tf.keras.layers.Conv2D(channels, 1)
        self._v = tf.keras.layers.Conv2D(channels, 1)
        # zero-initialised output projection => attention starts as identity
        self._proj = tf.keras.layers.Conv2D(
            channels, 1, kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, x, context=None):
        if context is None:
            context = x
        shp = tf.shape(x)
        b, h, w = shp[0], shp[1], shp[2]
        n = h * w
        # static guard for very wide grids
        hs, ws = x.shape[1], x.shape[2]
        if hs is not None and ws is not None and hs * ws > self._max_tokens:
            return x

        q = self._q(self._gn_q(x))
        k = self._k(self._gn_k(context))
        v = self._v(self._gn_v(context))

        q = tf.reshape(q, [b, n, self._channels])
        k = tf.reshape(k, [b, n, self._channels])
        v = tf.reshape(v, [b, n, self._channels])

        scale = 1.0 / tf.math.sqrt(tf.cast(self._channels, tf.float32))
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * scale, axis=-1)
        out = tf.matmul(attn, v)
        out = tf.reshape(out, [b, h, w, self._channels])
        return x + self._proj(out)


# ---------------------------------------------------------------------------
# ARA network (paper Fig. 3)
# ---------------------------------------------------------------------------

class ARANetwork(Model):
    """Attentive Residual Autoencoder for frequency-space channel denoising.

    Input / output : [batch, H(space=num_rx_ant), W(freq=subcarriers), 2]
    real/imag stacked in the last dimension. The network predicts a residual
    that is added to the input; the residual projection is zero-initialised so
    that an *untrained* network reproduces the classical LS+linear estimate
    exactly (identity refinement), and training only improves upon it.
    """

    def __init__(self, base_channels=16, dropout_rate=0.0,
                 use_attention=True, **kwargs):
        super().__init__(**kwargs)
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self._use_attention = use_attention

        # stem
        self._in_conv = tf.keras.layers.Conv2D(c1, 3, padding="same")

        # encoder level 1
        self._enc1_a = ResModule(c1, dropout_rate)
        self._enc1_b = ResModule(c1, dropout_rate)
        self._att1 = SpatialAttention(c1) if use_attention else None
        self._down1 = ResDown(c2, dropout_rate)

        # encoder level 2
        self._enc2_a = ResModule(c2, dropout_rate)
        self._att2 = SpatialAttention(c2) if use_attention else None
        self._down2 = ResDown(c3, dropout_rate)

        # bottleneck
        self._bott = ResModule(c3, dropout_rate)

        # decoder level 2
        self._up2 = ResUp(c2, dropout_rate)
        self._catt2 = SpatialAttention(c2) if use_attention else None
        self._dec2 = ResModule(c2, dropout_rate)

        # decoder level 1
        self._up1 = ResUp(c1, dropout_rate)
        self._catt1 = SpatialAttention(c1) if use_attention else None
        self._dec1 = ResModule(c1, dropout_rate)

        self._out_gn = _groupnorm(c1)
        # zero-init residual read-out => identity at initialisation
        self._out_conv = tf.keras.layers.Conv2D(
            2, 3, padding="same",
            kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, x, training=False):
        # pad H, W to a multiple of 4 (two down-sampling levels)
        shp = tf.shape(x)
        h, w = shp[1], shp[2]
        ph = (-h) % 4
        pw = (-w) % 4
        xin = tf.pad(x, [[0, 0], [0, ph], [0, pw], [0, 0]])

        s = self._in_conv(xin)

        # encoder
        e1 = self._enc1_a(s, training=training)
        e1 = self._enc1_b(e1, training=training)
        if self._att1 is not None:
            e1 = self._att1(e1)
        d1 = self._down1(e1, training=training)

        e2 = self._enc2_a(d1, training=training)
        if self._att2 is not None:
            e2 = self._att2(e2)
        d2 = self._down2(e2, training=training)

        # bottleneck
        b = self._bott(d2, training=training)

        # decoder level 2 (skip + cross-attention to matching encoder features)
        u2 = self._up2(b, target_hw=(tf.shape(e2)[1], tf.shape(e2)[2]),
                       training=training)
        if self._catt2 is not None:
            u2 = self._catt2(u2, context=e2)
        u2 = self._dec2(u2 + e2, training=training)

        # decoder level 1
        u1 = self._up1(u2, target_hw=(tf.shape(e1)[1], tf.shape(e1)[2]),
                       training=training)
        if self._catt1 is not None:
            u1 = self._catt1(u1, context=e1)
        u1 = self._dec1(u1 + e1, training=training)

        res = self._out_conv(tf.nn.relu(self._out_gn(u1)))

        # crop padding and add residual to the input estimate
        res = res[:, :h, :w, :]
        return x + res


# ---------------------------------------------------------------------------
# Sionna-compatible channel estimator
# ---------------------------------------------------------------------------

class ARAChannelEstimator(PUSCHLSChannelEstimator):
    r"""LS + linear-interpolation channel estimator refined by the ARA network.

    Drop-in replacement for ``PUSCHLSChannelEstimator`` inside a Sionna
    ``PUSCHReceiver``. Stages 1-2 (LS despreading + linear interpolation) are
    performed by the parent class; the resulting full-grid estimate is then
    denoised by :class:`ARANetwork` over the frequency-space plane
    ``(num_rx_ant, num_subcarriers)``.

    Input
    -----
    [y, no] : see ``PUSCHLSChannelEstimator``.

    Output
    ------
    h_hat  : [batch, num_rx, num_rx_ant, num_tx, num_streams,
              num_ofdm_symbols, num_effective_subcarriers], tf.complex
    err_var: broadcastable to ``h_hat``, tf.float
    """

    def __init__(self, resource_grid, dmrs_length, dmrs_additional_position,
                 num_cdm_groups_without_data,
                 base_channels=16, dropout_rate=0.0, use_attention=True,
                 **kwargs):
        super().__init__(resource_grid=resource_grid,
                         dmrs_length=dmrs_length,
                         dmrs_additional_position=dmrs_additional_position,
                         num_cdm_groups_without_data=num_cdm_groups_without_data,
                         interpolation_type="lin",
                         dtype=tf.complex64,
                         **kwargs)
        self._ara = ARANetwork(base_channels=base_channels,
                               dropout_rate=dropout_rate,
                               use_attention=use_attention)

    def _denoise(self, h_hat, training=False):
        # h_hat: [B, R, A, T, S, O, F]  (A=rx_ant=space, F=subcarrier=freq)
        shp = tf.shape(h_hat)
        B, R, A, T, S, O, F = [shp[i] for i in range(7)]

        # bring (A, F) to the last two axes: [B, R, T, S, O, A, F]
        hp = tf.transpose(h_hat, perm=[0, 1, 3, 4, 5, 2, 6])
        # merge everything except the frequency-space plane into the batch
        hp = tf.reshape(hp, [-1, A, F])
        # complex -> (real, imag) channels
        x = tf.stack([tf.math.real(hp), tf.math.imag(hp)], axis=-1)  # [N,A,F,2]

        y = self._ara(x, training=training)                          # [N,A,F,2]

        out = tf.complex(y[..., 0], y[..., 1])                       # [N,A,F]
        out = tf.reshape(out, [B, R, T, S, O, A, F])
        out = tf.transpose(out, perm=[0, 1, 5, 2, 3, 4, 6])          # [B,R,A,T,S,O,F]
        return out

    def call(self, inputs, training=False):
        h_hat, err_var = super().call(inputs)
        h_hat = self._denoise(h_hat, training=training)
        return h_hat, err_var
