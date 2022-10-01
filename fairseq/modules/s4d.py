import math
from typing import Dict, Optional, Tuple

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import opt_einsum as oe
from fairseq.incremental_decoding_utils import with_incremental_state
contract = oe.contract

logger = logging.getLogger(__name__)


try: # Try pykeops
    import pykeops
    from pykeops.torch import Genred
    has_pykeops = True
    logger.info("Pykeops installation found.")

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
        return tensors

    def cauchy_conj(v, z, w):
        """ Pykeops version """
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'

        cauchy_mult = Genred(
            f'ComplexDivide({expr_num}, {expr_denom})',
            [
                'v = Vj(2)',
                'z = Vi(2)',
                'w = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2*cauchy_mult(v, z, w, backend='GPU')
        return _r2c(r)

    def log_vandermonde(v, x, L):
        expr = 'ComplexMult(v, ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'v = Vj(2)',
                'x = Vj(2)',
                'l = Vi(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        v, x, l = _broadcast_dims(v, x, l)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(v, x, l, backend='GPU')
        return 2*_r2c(r).real

    def log_vandermonde_transpose(u, v, x, L):
        """
        u: ... H L
        v: ... H N
        x: ... H N
        Returns: ... H N
        V = Vandermonde(a, L) : (H N L)
        contract_L(V * u * v)
        """
        expr = 'ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'u = Vj(2)',
                'v = Vi(2)',
                'x = Vi(2)',
                'l = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        u, v, x, l = _broadcast_dims(u, v, x, l)
        u = _c2r(u)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(u, v, x, l, backend='GPU')
        return _r2c(r)

except ImportError:
    has_pykeops = False
    # Vandermonde functions
    logger.warning(
        "Falling back on slow Vandermonde kernel. Install pykeops for improved memory efficiency."
    )
    def log_vandermonde(v, x, L):
        """
        v: (..., N)
        x: (..., N)
        returns: (..., L) \sum v x^l
        """
        vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
        vandermonde_prod = contract('... n, ... n l -> ... l', v, vandermonde_matrix) # (... L)
        return 2*vandermonde_prod.real

    def log_vandermonde_transpose(u, v, x, L):
        vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
        vandermonde_prod = contract('... l, ... n, ... n l -> ... n', u.to(x), v.to(x), vandermonde_matrix) # (... L)
        return vandermonde_prod


_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
_resolve_conj = lambda x: x.conj().resolve_conj()


def dplr(scaling, N, rank=1, H=1, dtype=torch.float, real_scale=1.0, imag_scale=1.0, random_real=False, random_imag=False, normalize=False, diagonal=True, random_B=False):
    assert dtype == torch.float or torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)
    if random_real:
        real_part = torch.rand(H, N//2)
    else:
        real_part = .5 * torch.ones(H, N//2)
    if random_imag:
        imag_part = N//2 * torch.rand(H, N//2)
    else:
        imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    real_part = real_scale * real_part
    if scaling == 'random':
        imag_part = torch.randn(H, N//2)
    elif scaling == 'real':
        imag_part = 0 * imag_part
        real_part = 1 + repeat(torch.arange(N//2), 'n -> h n', h=H)
    elif scaling in ['linear', 'lin']:
        imag_part = pi * imag_part
    elif scaling in ['inverse', 'inv']: # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    elif scaling in ['inverse2', 'inv2']:
        imag_part = 1/pi * N * (N/(1+imag_part)-1)
    elif scaling in ['quadratic', 'quad']:
        imag_part = 1/pi * (1+2*imag_part)**2

    else: raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part

    # Initialize B
    if random_B:
        B = torch.randn(H, N//2, dtype=dtype)
    else:
        B = torch.ones(H, N//2, dtype=dtype)

    if normalize:
        norm = -B/w # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True)  # Variance with a random C vector
        B = B / zeta**.5

    P = torch.randn(rank, H, N//2, dtype=dtype)
    if diagonal: P = P * 0.0
    V = torch.eye(N, dtype=dtype)[: :N//2] # Only used in testing
    V = repeat(V, 'n m -> h n m', h=H)

    return w, P, B, V


def ssm(measure, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization
    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """
    args = measure.split("-")
    assert args[0] == "diag" and len(args) > 1
    scaling = args[1]
    w, P, B, V = dplr(scaling=scaling, N=N, rank=R, H=H, diagonal=True, **ssm_args)
    return w, P, B, V


combinations = {
    'hippo': ['legs', 'fourier'],
    'diag': ['diag-inv', 'diag-lin'],
    'all': ['legs', 'fourier', 'diag-inv', 'diag-lin'],
}

def combination(measures, N, R, S, **ssm_args):
    if isinstance(measures, str):
        measures = combinations[measures] if measures in combinations else [measures]

    assert S % len(measures) == 0, f"{S} independent trainable SSM copies must be multiple of {len(measures)} different measures"
    w, P, B, V = zip(
        *[ssm(measure, N, R, S // len(measures), **ssm_args) for measure in measures]
    )
    w = torch.cat(w, dim=0) # (S N)
    P = torch.cat(P, dim=1) # (R S N)
    B = torch.cat(B, dim=0) # (S N)
    V = torch.cat(V, dim=0) # (S N N)
    return w, P, B, V


""" Misc functional utilities """

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i
    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


class SSKernelDiag(nn.Module):
    """Version using (complex) diagonal state matrix (S4D)"""

    def __init__(
        self,
        A, B, C, log_dt,
        L=None,
        disc='bilinear',
        real_type='exp',
        lr=None,
        bandlimit=None,
    ):

        super().__init__()
        self.L = L
        self.disc = disc
        self.bandlimit = bandlimit
        self.real_type = real_type

        # Rank of low-rank correction
        assert A.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = A.size(-1)
        assert A.size(-2) == B.size(-2) # Number of independent SSMs trained
        assert self.H % A.size(-2) == 0
        self.n_ssm = A.size(-2)
        self.repeat = self.H // A.size(0)

        self.channels = C.shape[0]
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        # Register parameters
        if lr is None or isinstance(lr, float): lr_dict = {}
        else: lr_dict, lr = lr, None

        self.register_parameter("log_dt", nn.Parameter(log_dt))
        self.register_parameter("B", nn.Parameter(_c2r(B)))
        self.register_parameter("inv_A_real", nn.Parameter(self._A_init(A.real)))
        self.register_parameter("A_imag", nn.Parameter(A.imag))

    def _A_init(self, A_real):
        A_real = torch.clamp(A_real, max=-1e-4)
        if self.real_type == 'none':
            return -A_real
        elif self.real_type == 'exp':
            return torch.log(-A_real) # Some of the HiPPO methods have real part 0
        elif self.real_type == 'relu':
            return -A_real
        elif self.real_type == 'sigmoid':
            return torch.logit(-A_real)
        elif self.real_type == 'softplus':
            return torch.log(torch.exp(-A_real)-1)
        else: raise NotImplementedError

    def _A(self):
        # Get the internal A (diagonal) parameter
        if self.real_type == 'none':
            A_real = -self.inv_A_real
        elif self.real_type == 'exp':
            A_real = -torch.exp(self.inv_A_real)
        elif self.real_type == 'relu':
            # JAX version seems to NaN if you alloA 0's, although this code Aas fine Aithout it
            A_real = -F.relu(self.inv_A_real)-1e-4
        elif self.real_type == 'sigmoid':
            A_real = -F.sigmoid(self.inv_A_real)
        elif self.real_type == 'softplus':
            A_real = -F.softplus(self.inv_A_real)
        else: raise NotImplementedError
        A = A_real + 1j * self.A_imag
        return A

    def forward(self, L, state=None, rate=1.0, u=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length
        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """

        dt = torch.exp(self.log_dt) * rate # (H)
        C = _r2c(self.C) # (C H N)
        A = self._A() # (H N)

        B = _r2c(self.B)
        B = repeat(B, 't n -> 1 (v t) n', v=self.repeat)

        if self.bandlimit is not None:
            freqs = dt[:, None] / rate * A.imag.abs() / (2*math.pi) # (H, N)
            mask = torch.where(freqs < self.bandlimit * .5, 1, 0)
            C = C * mask

        # Incorporate dt into A
        A = repeat(A, 't n -> (v t) n', v=self.repeat)
        dtA = A * dt.unsqueeze(-1)  # (H N)

        # Augment B with state
        if state is not None:
            s = state / dt.unsqueeze(-1)
            if self.disc == 'bilinear':
                s = s * (1. + dtA/2)
            elif self.disc == 'zoh':
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.)
            B = torch.cat([s, B], dim=-3) # (1+B H N)

        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)
        if self.disc == 'zoh':
            # Power up
            C = C * (torch.exp(dtA)-1.) / A
            K = log_vandermonde(C, dtA, L) # (H L)
        elif self.disc == 'bilinear':
            C = C * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1) # or * dtA / A
            dA = (1. + dtA/2) / (1. - dtA/2)
            K = log_vandermonde(C, dA.log(), L)
        elif self.disc == 'dss':
            # Implementation from DSS meant for case when real eigenvalues can be positive
            P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device) # [H N L]
            A_gt_0 = A.real > 0                                      # [N]
            if A_gt_0.any():
                with torch.no_grad():
                    P_max = dtA * (A_gt_0 * (L-1))                   # [H N]
                P = P - P_max.unsqueeze(-1)                          # [H N L]
            S = P.exp()                                              # [H N L]

            dtA_neg = dtA * (1 - 2*A_gt_0)                           # [H N]
            num = dtA_neg.exp() - 1                                  # [H N]
            den = (dtA_neg * L).exp() - 1                            # [H N]

            # Inline reciprocal function for DSS logic
            x = den * A
            x_conj = _resolve_conj(x)
            r = x_conj / (x*x_conj + 1e-7)

            C = C * num * r             # [C H N]
            K = contract('chn,hnl->chl', C, S).float()
        else: assert False, f"{self.disc} not supported"

        K = K.view(-1, self.channels, self.H, L) # (1+B C H L)
        if state is not None:
            K_state = K[:-1, :, :, :] # (B C H L)
        else:
            K_state = None
        K = K[-1, :, :, :] # (C H L)
        return K, K_state

    def _setup_step(self):
        # These methods are organized like this to be compatible with the NPLR kernel interface
        dt = torch.exp(self.log_dt) # (H)
        B = _r2c(self.B) # (H N)
        C = _r2c(self.C) # (C H N)
        self.dC = C
        A = self._A() # (H N)

        # Incorporate dt into A
        dtA = A * dt.unsqueeze(-1)  # (H N)
        if self.disc == 'zoh':
            self.dA = torch.exp(dtA) # (H N)
            self.dB = B * (torch.exp(dtA)-1.) / A # (C H N)
        elif self.disc == 'bilinear':
            self.dA = (1. + dtA/2) / (1. - dtA/2)
            self.dB = B * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1) # or * dtA / A

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state

    def forward_state(self, u, state):
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous() # (B H L)
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v

        # todo: no masking on u?

        return next_state


class SSKernel(nn.Module):
    """Wrapper around SSKernel parameterizations.
    The SSKernel is expected to support the interface
    forward()
    default_state()
    _setup_step()
    step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=None,
        measure="diag-lin",
        rank=1,
        channels=1,
        dt_min=0.001,
        dt_max=0.1,
        deterministic=False,
        lr=None,
        n_ssm=None,
        verbose=False,
        measure_args={},
        **kernel_args,
    ):
        """State Space Kernel which computes the convolution kernel $\\bar{K}$
        H: Number of independent SSM copies; controls the size of the model. Also called d_model in the config.
        N: State size (dimensionality of parameters A, B, C). Also called d_state in the config. Generally shouldn't need to be adjusted and doens't affect speed much.
        L: Maximum length of convolution kernel, if known. Should work in the majority of cases even if not known.
        measure: Options for initialization of (A, B). For NPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin)
        rank: Rank of low-rank correction for NPLR mode. Needs to be increased for measure "legt"
        channels: C channels turns the SSM from a 1-dim to C-dim map; can think of it having C separate "heads" per SSM. This was partly a feature to make it easier to implement bidirectionality; it is recommended to set channels=1 and adjust H to control parameters instead
        dt_min, dt_max: min and max values for the step size dt (\Delta)
        mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D; 'slow' is a dense version for testing
        n_ssm: Number of independent trainable (A, B) SSMs, e.g. n_ssm=1 means all A/B parameters are tied across the H different instantiations of C. n_ssm=None means all H SSMs are completely independent. Generally, changing this option can save parameters but doesn't affect performance or speed much. This parameter must divide H
        lr: Passing in a number (e.g. 0.001) sets attributes of SSM parameers (A, B, dt). A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        """
        super().__init__()
        self.N = N
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm if n_ssm is not None else H
        self.verbose = verbose
        self.kernel_args = kernel_args

        # Generate dt
        if deterministic:
            log_dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), H))
        else:
            log_dt = torch.rand(self.H, dtype=dtype) * (
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)

        # Compute the preprocessed representation
        w, P, B, V = combination(measure, self.N, rank, self.n_ssm, **measure_args)

        # Broadcast C to have H channels
        if deterministic:
            C = torch.zeros(channels, self.H, self.N, dtype=cdtype)
            C[:, :, :1] = 1.
            C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C) # V^* C
        else:
            C = torch.randn(channels, self.H, self.N//2, dtype=cdtype)

        # Broadcast other parameters to have n_ssm copies
        assert self.n_ssm % B.size(-2) == 0 \
                and self.n_ssm % P.size(-2) == 0 \
                and self.n_ssm % w.size(-2) == 0
        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
        w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()
        C = C.contiguous()

        if not measure.startswith("diag"):
            logger.warning("Diagonal kernel (S4D) activated but initialization is not intended for S4D. Set `measure` to 'diag-lin', 'diag-inv', or 'diag-legs' for the main variants, or 'diag' for a combination of S4D-Lin and S4D-Inv.")
        C = C * repeat(B, 't n -> (v t) n', v=H//self.n_ssm)
        self.kernel = SSKernelDiag(
            w, B, C, log_dt, L=L,
            lr=lr,
            **kernel_args,
        )

    def forward(self, state=None, L=None, rate=None):
        return self.kernel(state=state, L=L, rate=rate)

    # @torch.no_grad()  #todo: ?
    def forward_state(self, u, state):
        """ Forward the state through a sequence, i.e. computes the state after passing chunk through SSM
        state: (B, H, N)
        u: (B, H, L)
        Returns: (B, H, N)
        """

        if hasattr(self.kernel, "forward_state"):
            return self.kernel.forward_state(u, state)

        dA, dB = self.kernel._setup_state() # Construct dA, dB matrices
        # dA, dB = self.kernel.dA, self.kernel.dB # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj: state = _conj(state)

        v = contract('h n, b h l -> b h n l', dB, u.flip(-1)) # dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2)
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj: next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_step(self, **kwargs):
        # This method is intended to be private so that setting up an S4 module with
        # ```
        # if hasattr(module, 'setup_step'): module.setup_step()
        # ```
        # will not trigger this method multiple times
        self.kernel._setup_step(**kwargs)

    def step(self, u, state, **kwargs):
        y, state = self.kernel.step(u, state, **kwargs)
        return y, state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


@with_incremental_state
class S4D(nn.Module):
    def __init__(
            self,
            embed_dim,
            ndim=64,
            bidirectional=False,
            l_max=None,
            channels=1,
            # Arguments for position-wise feedforward components
            transposed=True,
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided
        See the class SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"
        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            logger.info(f"Constructing S4 (H, N, L) = ({embed_dim}, {ndim}, {l_max})")

        self.d_model = embed_dim
        self.H = embed_dim
        self.N = ndim
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels  # todo: ?
        self.transposed = transposed

        # todo: ?
        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=channels, verbose=verbose, **kernel_args)
        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(self, u, padding_mask=None, state=None, rate=1.0, incremental_state=None, lengths=None, **kwargs):
        """
        Input shape u: Time x Batch x Channel (L B H)
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,

        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing
        Returns: same shape as u
        """
        # if not self.transposed: u = u.transpose(-1, -2)
        u = u.permute(1, 2, 0)  # (B H L)
        L = u.size(-1)

        # Mask out padding tokens
        # if isinstance(lengths, int):
        #     if lengths != L:
        #         lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
        #     else:
        #         lengths = None
        # if lengths is not None:
        #     assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
        #     mask = torch.where(torch.arange(L, device=lengths.device) < lengths[:, None, None], 1., 0.)
        #     u = u * mask

        if padding_mask is not None:
            u = u * (1.0 - padding_mask.unsqueeze(1).type_as(u))

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state = self.kernel(L=L_kernel, rate=rate, state=state)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))

        k_f = torch.fft.rfft(k, n=L_kernel+L) # (C H L)
        u_f = torch.fft.rfft(u, n=L_kernel+L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=L_kernel+L)[..., :L] # (B C H L)  # todo: ?

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)



        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)
        return y

    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx)
        next_state = self.kernel.forward_state(x, hx)  # (B, H, N)


    def one_step(self, u,  state=None):
        """ Step one time step as a recurrent model. Intended to be used during validation.
        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)
        # y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, 'b c h -> b (c h)')
        # y = self.activation(y)
        # if self.transposed:
        #     y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        # else:
        #     y = self.output_linear(y)
        # 1 x B x H, B x H x N
        return y.unsqueeze(1), next_state


