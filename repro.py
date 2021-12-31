"""
binary dilation in numba cuda
>> python cuda_binary_dilate_3d.py 512 512 512 1
>>> runtime: 0.001s on a Nvidia TitanXP. 
"""
import numba
import numpy as np
import time
from numba import cuda 


@cuda.jit("void(uint8[:,:,:],uint8[:,:,:],uint8[:,:,:])")
def cuda_binary_dilate_u8(vol, out, kern):
    z,y,x = cuda.grid(3)
    d,h,w = vol.shape
    a,b,c = kern.shape
    pa,pb,pc = a//2,b//2,c//2
    if z >= 0 and z < d and y >= 0 and y < h and x >= 0 and x < w:
        out[z,y,x] = False
        
        #put kern in shared memory?
        shared = cuda.shared.array((3,3,3),dtype=numba.boolean)
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    shared[i,j,k] = kern[i,j,k]
        
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    zz = z+i-pa
                    yy = y+j-pb
                    xx = x+k-pc
                    if zz < 0 or zz >= d or yy < 0 or yy >= h or xx < 0 or xx >= w: 
                        continue
                    if vol[zz,yy,xx] and shared[i,j,k]:
                       out[z,y,x] = True




def binary_dilate_u8_cuda(vol, kern, iterations=1):
    sizes = vol.shape 
    block_dim = (4,4,4)
    #stride
    sizes = [v for v in sizes]
    grid_dim = tuple(int(np.ceil(a/b)) for a, b in zip(sizes, block_dim))

    # non inplace
    ycu = cuda.device_array(shape=vol.shape, dtype=np.uint8)
    a,b = ycu, vol 
    for i in range(iterations):
         a,b = b,a
         cuda_binary_dilate_u8[grid_dim, block_dim](a, b, kern)
    return b


def test(d=256,h=256,w=256,niter=1):
    vol = np.random.randn(d,h,w) > 1
    kern = np.random.randn(3,3,3) > 1

    vol = cuda.to_device(vol)
    kern = cuda.to_device(kern)
    
    t1 = time.time()
    for i in range(10):
        binary_dilate_u8_cuda(vol, kern, niter)
    t2 = time.time()
    
    print(f' runtime: {(t2-t1)/10}') # i measure 1ms


if __name__ == '__main__':
    test()
