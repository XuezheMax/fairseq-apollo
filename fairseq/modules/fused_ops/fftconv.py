from typing import Tuple, Optional

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx

from fairseq.mega2_extension.ops import rfft, fftconv_fwd, fftconv_bwd


class FFTConvFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor, k: torch.Tensor, bidirectional: bool = False) -> torch.Tensor:
        L = x.size(-1)
        if bidirectional or not x.is_cuda or L < 32 or L > 8192:
            y, x_f, k_f = _fftconv_fwd(x, k, bidirectional)
        else:
            N = L if (L & (L - 1)) == 0 else (1 << L.bit_length())
            if k.dtype == torch.float32 or k.dtype == torch.float64:
                k_f = torch.fft.rfft(k, 2 * N)
            else:
                k_f = rfft(k, False)
            y, x_f = fftconv_fwd(x, k_f)
        ctx.save_for_backward(x_f, k_f)
        ctx.k_dtype = k.dtype  # k_dtype is not a torch.Tensor
        ctx.bidirectional = bidirectional
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        L = y_grad.size(-1)
        x_f, k_f = ctx.saved_tensors
        k_dtype = ctx.k_dtype
        bidirectional = ctx.bidirectional

        if bidirectional or not y_grad.is_cuda or L < 32 or L > 8192:
            x_grad, k_grad = _fftconv_bwd(y_grad, x_f, k_f, bidirectional)
        else:
            x_grad, k_grad = fftconv_bwd(y_grad, x_f, k_f, k_dtype)

        return x_grad, k_grad, None


fftconv = FFTConvFunc.apply


# @torch.jit.script
def _fftconv_fwd(x: torch.Tensor, k: torch.Tensor,
                 bidirectional: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L: int = x.size(-1)
    K: int = k.size(-1)
    if bidirectional:
        assert L * 2 == K
    else:
        assert L == K

    N: int = 2 * L
    N = N if (N & (N - 1)) == 0 else (1 << N.bit_length())
    dtype = x.dtype

    if dtype == torch.float16 or dtype == torch.bfloat16:
        x = x.float()

    x_f = torch.fft.rfft(x, n=N)
    k_f = torch.fft.rfft(k, n=N, norm="forward")
    y_f = x_f * k_f
    y = torch.fft.irfft(y_f, n=N, norm="forward")[..., :L]

    if dtype == torch.float16 or dtype == torch.bfloat16:
        y = y.to(dtype)

    return y, x_f, k_f


# @torch.jit.script
def _fftconv_bwd(y_grad: torch.Tensor, x_f: torch.Tensor, k_f: torch.Tensor,
                 bidirectional: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    L: int = y_grad.size(-1)
    K: int = 2 * L if bidirectional else L
    N: int = 2 * L
    N = N if (N & (N - 1)) == 0 else (1 << N.bit_length())

    dtype = y_grad.dtype
    if dtype == torch.float16 or dtype == torch.bfloat16:
        y_grad = y_grad.float()

    y_grad_f = torch.fft.rfft(y_grad.flip(dims=(-1,)), n=N)
    x_grad_f = y_grad_f * k_f
    k_grad_f = y_grad_f * x_f
    x_grad = torch.fft.irfft(x_grad_f, n=N, norm="forward")[..., :L].flip(dims=(-1,))
    k_grad = torch.fft.irfft(k_grad_f.sum(dim=0), n=N)[..., :K].flip(dims=(-1,))

    if dtype == torch.float16 or dtype == torch.bfloat16:
        x_grad = x_grad.to(dtype)

    return x_grad, k_grad
