import torch 
from numba import cuda, jit, float32
import time
# from causal_attention_cuda import CausalAttnCuda

@cuda.jit
def efficient_causal_attention_cuda(x, y, z, n, d1, d2, accum_mat, rets):
    bx = cuda.blockIdx.x # B
    by = cuda.blockIdx.y # d1
    bz = cuda.blockIdx.z # d2
    tx = cuda.threadIdx.x # n
    xx = x[bx, tx, by]
    yy = y[bx, tx, by]
    zz = z[bx, tx, bz]
    accum_mat[bx][tx][by][bz] = yy * zz
    cuda.syncthreads()
    if tx == 0:
        for i in range(1, n):
            accum_mat[bx][i][by][bz] += accum_mat[bx][i-1][by][bz]
    cuda.syncthreads()
    # for i in range(d1):
    #     tmp = 0.
    #     tmp += x[bx, tx, by] * accum_mat[bx][tx][i][bz]
    if by == 0:
        tmp = 0
        for i in range(d1):
            tmp += x[bx][tx][i] * accum_mat[bx][tx][i][bz]
        rets[bx][tx][bz] = tmp / (tx + 1.)
    # cuda.syncthreads()
    # rets[bx][tx][bz] += xx * accum_mat[bx][tx][by][bz] / (tx + 1.)

def get_devicendarray(t):
    assert t.type() == 'torch.cuda.FloatTensor'
    ctx = cuda.cudadrv.driver.driver.get_context()
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
    return cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], numpy.dtype('float32'), 
                                                  gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)   

def efficient_causal_attention(x, y, z):
    """
    efficient causal attention operation
    Args:
        x (Tensor): Tensor with shape `(batch, n, d1)`
        y (Tensor): Tensor with shape `(batch, n, d1)`
        z (Tensor): Tensor with shape '(batch, n, d2)`

    return:
    """
    n = x.size(1)
    rets = []
    accum_mat = 0
    for i in range(n):
        xx = x[:, i:i + 1] # B x 1 x d1
        yy = y[:, i:i + 1] # B x 1 x d1
        zz = z[:, i:i + 1] # B x 1 x d2

        # B x d1 x d2
        accum_mat = accum_mat + torch.bmm(yy.transpose(1, 2), zz)
        # B x 1 x d2
        rets.append(torch.bmm(xx, accum_mat).div(i + 1.))
    # B x N x d2
    return torch.cat(rets, dim=1), accum_mat

def efficient_causal_attention_v2(x, y, z):
    """
    efficient causal attention operation
    Args:
        x (Tensor): Tensor with shape `(batch, n, d1)`
        y (Tensor): Tensor with shape `(batch, n, d1)`
        z (Tensor): Tensor with shape '(batch, n, d2)`

    return:
    """
    B, n, d1 = x.size()
    d2 = z.size(-1)
    rets = []
    accum_mat = 0
    sum_mat = torch.bmm(y.view(B*n, d1, 1), z.view(B*n, 1, d2)).view(B, n, d1, d2)
    accum_mat = torch.cumsum(sum_mat, dim=1)
    length_div = torch.arange(1, n+1, device=x.device).unsqueeze(0).unsqueeze(2).contiguous()
    rets = torch.bmm(x.view(B*n, 1, d1), accum_mat.view(B*n, d1, d2)) 
    return rets.view(B, n, d2) / length_div, accum_mat
    # for i in range(n):
    #     xx = x[:, i:i + 1] # B x 1 x d1
    #     yy = y[:, i:i + 1] # B x 1 x d1
    #     zz = z[:, i:i + 1] # B x 1 x d2

    #     # B x d1 x d2
    #     accum_mat = accum_mat + torch.bmm(yy.transpose(1, 2), zz)
    #     # B x 1 x d2
    #     rets.append(torch.bmm(xx, accum_mat).div(i + 1.))
    # # B x N x d2
    # return torch.cat(rets, dim=1), accum_mat


class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()
	# end

	def forward(self, x, y, z):
		return CausalAttnCuda.apply(x, y, z)
	# end
# end

if __name__ == "__main__":

    torch.cuda.set_device(1)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    B = 100
    n = 40
    d1 = 512
    d2 = 512
    times = 100
    # maxd = max(d1, d2)
    x = torch.randn(B, n, d1).cuda()
    y = torch.randn(B, n, d1).cuda()
    z = torch.randn(B, n, d2).cuda()
    # net = Network().cuda()
    # rets = net(x, y, z)
    
    # print(accum_mat)
    # print(x, y, z)
    start = time.time()
    for i in range(times):
        ret, acc = efficient_causal_attention(x, y, z)
    print((time.time() - start) / times)
    # for i in range(times):
    #     ret2, acc2 = efficient_causal_attention_v2(x, y, z)
    
    # print(torch.sum((ret2 - ret) ** 2))
    exit()
    
    # accum_mat = torch.zeros(B, n, d1, d2).cuda()
    # rets = torch.zeros(B, n, d2).cuda()
    # start = time.time()
    # x.requires_grad = True
    # for i in range(times):
    #     efficient_causal_attention_cuda[(B, d1, d2), n](
    #         get_devicendarray(x), 
    #         cuda.as_cuda_array(y), 
    #         cuda.as_cuda_array(z), 
    #         n,
    #         d1,
    #         d2,
    #         cuda.as_cuda_array(accum_mat),
    #         cuda.as_cuda_array(rets)
    #     )
    # # print((time.time() - start) / times)
    # print(accum_mat, acc)
    # print(torch.sum((accum_mat[:,-1, ...]-acc) ** 2))
    # print(accum_mat)
    # print(rets, ret)
    # print(torch.sum((rets - ret) ** 2))
    # print(torch.autograd.gradcheck(net, tuple([x, y, z]), 0.001), '<-- should be true')
    exit()
    # # print(rets, ret)