#!/usr/bin/env python

import torch

import cupy

kernel_causal_attention_Output1 = '''
    extern "C" __global__ void kernel_causal_attention_Output1( 
        const float* y, 
        const float* z, 
        const int n, 
        const int d1, 
        const int d2, 
        const int B,
        float* accum_mat
    ){
        int bx = blockIdx.x; 
        int by = blockIdx.y; 
        int bz = blockIdx.z; 
        int tx = threadIdx.x;
        accum_mat[bx*n*d1*d2+tx*d1*d2+d2*by+bz] = y[bx*n*d1+d1*tx+by] * z[bx*n*d2+d2*tx+bz];
    }
'''

kernel_causal_attention_Output2 = '''
    extern "C" __global__ void kernel_causal_attention_Output2( 
        const float* x,
        const int n, 
        const int d1, 
        const int d2, 
        const int B,
        float* accum_mat,
        float* rets
    ){
        int bx = blockIdx.x; 
        int by = blockIdx.y;
        int bz = blockIdx.z;
        float tmp = 0.;
        for (int i = 0; i < d1; ++i){
            tmp += x[bx*n*d1+d1*by+i] * accum_mat[bx*n*d1*d2+by*d1*d2+i*d2+bz];
        }
        rets[bx*n*d2+d2*by+bz] = tmp / (by + 1.);
    }
'''

kernel_causal_attention_GradInput1 = '''
    extern "C" __global__ void kernel_causal_attention_GradInput1(
        const int n, 
        const int d1, 
        const int d2, 
        const int B,
        float* accum_mat,
        const float* gradOutput,
        float* gradInputx
    ) {
        int bx = blockIdx.x; 
        int by = blockIdx.y;
        int bz = blockIdx.z;
        float tmp = 0.;
        for (int i = 0; i < d2; ++i){
            tmp += gradOutput[bx*n*d1+d1*by+i] * accum_mat[bx*n*d1*d2+by*d1*d2+bz*d2+i];
        }
        gradInputx[bx*n*d1+d1*by+i] = tmp;
    }
'''


@cupy.memoize(for_each_device=True)
def cunnex(strFunction):
    return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
# end

class CausalAttnCuda(torch.autograd.Function):
    @staticmethod
    def forward(self, input1, input2, input3):

        
        input1 = input1.requires_grad_()
        input2 = input2.requires_grad_()
        input3 = input3.requires_grad_()


        assert(input1.is_contiguous() == True)
        assert(input2.is_contiguous() == True)
        assert(input3.is_contiguous() == True)
        B, n, d1 = input1.size()
        d2 = input3.size(2)
        accum_mat = input1.new_zeros(B, n, d1, d2)
        rets = input1.new_zeros(B, n, d2)
        if input1.is_cuda == False:
            raise NotImplementedError()

        cunnex('kernel_causal_attention_Output1')(
            grid=tuple([B, d1, d2]),
            block=tuple([n]),
            args=[
                input2.data_ptr(),
                input3.data_ptr(),
                n,
                d1,
                d2,
                B,
                accum_mat.data_ptr()
            ]
        )
        sum_mat = accum_mat
        accum_mat = torch.cumsum(accum_mat, dim=1)
        self.save_for_backward(input1, input2, input3, accum_mat, sum_mat)
        cunnex('kernel_causal_attention_Output2')(
            grid=tuple([B, n, d2]),
            block=tuple([1]),
            args=[
                input1.data_ptr(),
                n,
                d1,
                d2,
                B,
                accum_mat.data_ptr(),
                rets.data_ptr()
            ]
        )
        return rets
    @staticmethod
    def backward(self, gradOutput):
        input1, input2, input3, accum_mat, sum_mat = self.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        B, n, d1 = input1.size()
        d2 = input3.size(2)
        gradInput1 = torch.bmm(gradOutput.view(B*n, 1, d2), accum_mat.transpose(2, 3).view(B*n, d2, d1))
        # length_div = torch.arange(1, n+1, device=input1.device).expand(B, n, 1).contiguous()
        length_div = torch.arange(1, n+1, device=input1.device).unsqueeze(0).unsqueeze(2).contiguous()
        gradInput1 = gradInput1.view(B, n, d1) / length_div
        reverse_cum_sum_x =  th.flip(th.cumsum(th.flip(input1, [1]), 1), [1])
        
        # cunnex('kernel_causal_attention_GradInput1')(
		# 		grid=tuple([B, n, d1]),
		# 		block=tuple([ 512, 1, 1 ]),
		# 		args=[ n, input1.data_ptr(), input2.data_ptr(), gradOutput.data_ptr(), gradInput1.data_ptr(), gradInput2.data_ptr() ]
		# 	)
        print(gradInput1)
        return gradInput1, input2, input3

