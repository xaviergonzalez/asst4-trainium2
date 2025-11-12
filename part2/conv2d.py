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
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    W_sbuf = nl.ndarray( # [out_channels, in_channels, filter_height, filter_width]
        shape=W.shape,
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    print()
    print(f"batch_size: {batch_size}, in_channels: {in_channels}, input_height: {input_height}, input_width: {input_width}")
    # pdb.set_trace()
    X_sbuf = nl.ndarray(
        shape=(in_channels, input_height, input_width),
        dtype=X.dtype,
        buffer=nl.sbuf,
    )
    # Process the images in batches
    nisa.dma_copy(src=W, dst=W_sbuf)
    for b in nl.sequential_range(batch_size):
    # for b in nl.affine_range(batch_size):
        nisa.dma_copy(
            src=X[b, :, :, :], dst=X_sbuf
        )  # Load the input image into SBUF
        # X_sbuf.reshape((in_channels, input_height * input_width)) # (in_channels, in_height * in_width)
        res_psum = nl.zeros((out_channels, out_height * out_width), dtype=X.dtype, buffer=nl.psum)
        for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):
        # raise RuntimeError("Please fill your implementation of computing convolution"
        #                    " of X[b] with the weights W and bias b, followed by a"
        #                    " maxpool and store the result in X_out[b]")' 
                # input_shifted = X[b, :, i:(i+out_height), j:(j+out_width)].reshape((in_channels, out_height * out_width)) # (in_channels, out_height * out_width)
                input_shifted = X_sbuf[:, i:(i+out_height), j:(j+out_width)].reshape((in_channels, out_height * out_width)) # (in_channels, out_height * out_width)
                res_psum += nl.transpose(nisa.nc_matmul(input_shifted, nl.transpose(W_sbuf[:, :, i, j]))) # (out_channels, out_height * out_width)
        res_psum = res_psum.reshape((out_channels, out_height, out_width)) # (out_channels, out_height, out_width)
        res_sbuf = nisa.tensor_copy(res_psum)
        nisa.dma_copy(src=res_sbuf, dst=X_out[b])

    return X_out

