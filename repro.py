"""
binary dilation in numba cuda
>> python cuda_binary_dilate_3d.py 512 512 512 1
>>> runtime: 0.001s on a Nvidia TitanXP.
"""
import numba
import numpy as np
import time
from numba import cuda
import pickle

# Whether to check output - reference version output's pickle not stored in git
# repo due to size.
CHECK_OUTPUT = False

# Hardcode dimension in kernel to allow for better compiler optimization
NKERN = 3


@cuda.jit
def cuda_binary_dilate_u8(vol, out, kern):
    z, y, x = cuda.grid(3)
    d, h, w = vol.shape
    pa, pb, pc = NKERN // 2, NKERN // 2, NKERN // 2

    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z

    if z < d and y < h and x < w:
        out[z, y, x] = False

        # put kern in shared memory?
        shared = cuda.shared.array((NKERN, NKERN, NKERN), dtype=numba.boolean)
        if tx < NKERN and ty < NKERN and tz < NKERN:
            shared[tx, ty, tz] = kern[tx, ty, tz]

        cuda.syncthreads()

        for i in range(NKERN):
            for j in range(NKERN):
                for k in range(NKERN):
                    zz = z + i - pa
                    yy = y + j - pb
                    xx = x + k - pc
                    if (zz < 0 or zz >= d or yy < 0 or yy >= h or xx < 0
                            or xx >= w):
                        continue
                    if vol[zz, yy, xx] and shared[i, j, k]:
                        out[z, y, x] = True


def binary_dilate_u8_cuda(vol, kern, iterations=1):
    sizes = vol.shape
    block_dim = (4, 4, 4)
    # stride
    sizes = [v for v in sizes]
    grid_dim = tuple(int(np.ceil(a / b)) for a, b in zip(sizes, block_dim))

    # non inplace
    ycu = cuda.device_array(shape=vol.shape, dtype=np.uint8)
    a, b = ycu, vol
    for i in range(iterations):
        a, b = b, a
        cuda_binary_dilate_u8[grid_dim, block_dim](a, b, kern)
    return b


def test(d=256, h=256, w=256, niter=1):
    np.random.seed(1)
    vol = np.random.randn(d, h, w) > 1
    kern = np.random.randn(3, 3, 3) > 1

    vol = cuda.to_device(vol)
    kern = cuda.to_device(kern)

    # Warm up / compile
    res = binary_dilate_u8_cuda(vol, kern, niter)

    if CHECK_OUTPUT:
        with open('reference.pickle', 'rb') as f:
            ref = pickle.load(f)

        np.testing.assert_equal(res.copy_to_host(), ref)

    t1 = time.time()
    for i in range(10):
        binary_dilate_u8_cuda(vol, kern, niter)
    cuda.synchronize()
    t2 = time.time()

    print(f" runtime: {(t2-t1)/10}")  # i measure 1ms


if __name__ == "__main__":
    import sys
    args = [int(a) for a in sys.argv[1:]]
    test(*args)
