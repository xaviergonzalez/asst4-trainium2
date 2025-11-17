import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

import pdb


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    pmax = nl.tile_size.pmax # 128
    n_tiles_c_in = in_channels // pmax
    n_tiles_c_out = out_channels // pmax

    # prepare sbuf arrays
    W_sbuf = nl.ndarray( #
        # shape=(pmax, pmax, filter_height, filter_width),
        shape = (pmax, pmax, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    X_sbuf = nl.ndarray(
        shape=(pmax, input_width), # (in_channels [cap at 128], input_width)
        dtype=X.dtype,
        buffer=nl.sbuf,
    )
    tile_sbuf = nl.ndarray(
                    shape=(pmax, pool_size, out_width), # (out_channels [cap at 128], pool_size, out_width)
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
    # load weight into SBUF
    # expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
    for tile_out in nl.affine_range(n_tiles_c_out):
        for tile_in in nl.affine_range(n_tiles_c_in):
            nisa.dma_copy(
                        src=W[tile_out * pmax:(tile_out + 1) * pmax, tile_in * pmax:(tile_in + 1) * pmax, :, :],
                        dst=W_sbuf[:, :, tile_out, tile_in, :, :],
                    )
            for i in nl.sequential_range(filter_height):
                for j in nl.sequential_range(filter_width):
                    W_sbuf[:, :, tile_out, tile_in, i, j] = nisa.dma_transpose(W_sbuf[:, :, tile_out, tile_in, i, j]) # switch to be (in_channels, out_channels)
                    
    for b in nl.sequential_range(batch_size): # the same as affine range with @nki.compiler.skip_middle_end_transformations
        for tile_out in nl.sequential_range(n_tiles_c_out): # tile over the out channels
            for ph in nl.sequential_range(out_pool_height):
                for p in nl.sequential_range(pool_size):
                    h = ph * pool_size + p
                    res_psum = nl.zeros((pmax, out_width), dtype=nl.float32, buffer=nl.psum) # (out_channels [cap at 128], out_width). Needs to be FP32. On PSUM.
                    for tile_in in nl.sequential_range(n_tiles_c_in):
                        # nisa.dma_copy(src=W[tile_out * pmax:(tile_out + 1) * pmax, tile_in * pmax:(tile_in + 1) * pmax, :, :], dst=W_sbuf)
                        # transpose W_sbuf here
                        # W_sbuf = nisa.dma_transpose(W_sbuf, axes = (3, 1, 2, 0))
                        for i in nl.sequential_range(filter_height):
                            # load current tiles into sbuf
                            nisa.dma_copy(
                                src=X[b, tile_in * pmax:(tile_in + 1) * pmax, (i+h), :], dst=X_sbuf
                            ) 
                            for j in nl.sequential_range(filter_width):
                                input_shifted = X_sbuf[:, j:(j+out_width)] # (in_channels [cap at 128], out_width)
                                # W_slice = W_sbuf[:, :, i, j] # (out_channels [cap at 128], in_channels [cap at 128])
                                # WT = nl.transpose(W_sbuf[:, :, tile_out, tile_in, i ,j]) # (in_channels [cap at 128], out_channels [cap at 128]), on PSUM
                                # WT_copy = nisa.tensor_copy(WT)
                                conv_res = nisa.nc_matmul(W_sbuf[:, :, tile_out, tile_in, i ,j], input_shifted) # (out_channels [cap at 128], out_width)
                                # conv_res = nisa.nc_matmul(W_sbuf[j, :, i, :], input_shifted) # (out_channels [cap at 128], out_width)
                                res_psum += conv_res # (out_channels [cap at 128], out_width), on PSUM
                    tile_sbuf[:, p, :] = nisa.tensor_copy(res_psum)
                    # res_sbuf = nisa.tensor_copy(res_psum) # (out_channels [cap at 128], out_width)
                    # nisa.dma_copy(src=res_sbuf, dst=tile_sbuf[:, p, :]) 
                # maxpool
                pre_maxpool = tile_sbuf.reshape((pmax, pool_size, out_width // pool_size, pool_size))
                post_maxpool = nisa.tensor_reduce(nl.max, pre_maxpool, axis=(1,3)) # (pmax, pool_width)
                bias_sbuf = nl.ndarray(
                        shape=(pmax, 1),
                        dtype=bias.dtype,
                        buffer=nl.sbuf,
                    )
                nisa.dma_copy(src=bias[tile_out * pmax:(tile_out + 1) * pmax], dst=bias_sbuf)
                post_maxpool = nisa.tensor_tensor(post_maxpool, bias_sbuf, op=nl.add) # (out_channels [cap at 128], out_width)
                nisa.dma_copy(src=post_maxpool, dst=X_out[b, tile_out * pmax:(tile_out + 1) * pmax, ph])

    return X_out # (batch_size, out_channels, out_pool_height, out_pool_width), on HBM

