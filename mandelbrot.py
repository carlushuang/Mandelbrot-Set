from __future__ import print_function
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from timeit import default_timer as timer
import numba as nb

ESCAPE_RADIUS=2**30
MAX_ITERATION=2048

# about 3:2 image
IMG_DPI=360
IMG_W_INCH=15
IMG_H_INCH=10
#
#RE_MIN=-2
#RE_MAX=1
#IM_MIN=-1
#IM_MAX=1

RE_MIN=np.float64(-0.74877)
RE_MAX=np.float64(-0.74871)
IM_MIN=np.float64(0.06506)
IM_MAX=np.float64(0.06510)


# imshow params
GAMMA=0.25
INTERPOLATION='bilinear'

#https://www.ibm.com/developerworks/community/blogs/jfp/entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en_us
#https://www.ibm.com/developerworks/community/blogs/jfp/entry/My_Christmas_Gift?lang=en

#@nb.jit(nb.complex64(nb.complex64,nb.complex64))
#@nb.jit(nopython=True)
@nb.vectorize(['complex128(complex128,complex128)'], nopython=True)
def madelbrot_func(z, c):
    return z*z + c

@nb.vectorize(['boolean(complex128,float64)'], nopython=True)
def mandelbrot_escape_p2(z, escape_radius_p2):
    return z.real*z.real+z.imag*z.imag > escape_radius_p2

# return n-log2(log(|Zn|)/log(N))
# n: iteration, |Zn|: nth z value length, N: radius
#  = n-log2(log(|Zn|))+log2(log(N))
#  = n-log2(0,5*log(Zn^2))+log2(log(N))
@nb.vectorize(['float64(int64,complex128,float64)'], nopython=True)
def continuous_color(i, z, log_radius):
    """
        http://math.unipa.it/~grim/Jbarrallo.PDF
    """
    return i-np.log2(0.5*np.log(z.real*z.real+z.imag*z.imag)) + log_radius


# return iteration

#@nb.guvectorize(['f8(c16,f8,i8)'],'')
@nb.vectorize(['float64(complex128,float64,int64)'],nopython=True)
def mandelbrot(c, escape_radius, max_iteration):
    c_plane = c
    escape_radius_p2 = np.float64(escape_radius*escape_radius)
    log_radius = np.log2(np.log(escape_radius))

    def z_func(z):
        return madelbrot_func(z, c_plane)
    def c_func(i, z):
        return continuous_color(i, z, log_radius)
        # return i
    def e_func(z, escape):
        return mandelbrot_escape_p2(z, escape)

    z = np.complex128(0)
    for i in range(max_iteration):
        # faster than abs(z) > escape_radius
        if e_func(z, escape_radius_p2):
            return c_func(i, z)
        z = z_func(z)
    return np.float64(0) # wrap aroubd for max iter, give a black color

#@nb.guvectorize(['void(complex128[:,:], float64[:,:])'],'(x,y)->(x,y)',nopython=True)

# use parallel=True, and prange(), to use multi-thread
@nb.jit(nopython=True,parallel=True)
def mandelbrot_region(c_region,depth):
    for j in nb.prange(c_region.shape[0]):
        for i in nb.prange(c_region.shape[1]):
            depth[j, i] = mandelbrot(c_region[j,i], np.float64(ESCAPE_RADIUS), np.int64(MAX_ITERATION))


@nb.jit(nopython=True,parallel=True)
def init_complex_region(res,ims,c_region):
    for j in nb.prange(c_region.shape[0]):
        for i in nb.prange(c_region.shape[1]):
            c_region[j,i] = res[i] + ims[j]*1j

def main():
    fig, ax = plt.subplots(figsize=(IMG_W_INCH, IMG_H_INCH), dpi=IMG_DPI)
    ax.set_axis_off()
    ax.set_position([0.,0.,1.,1.])
    ax.margins(0)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    fig.set_frameon(False)
    
    res = np.linspace(RE_MIN, RE_MAX, IMG_W_INCH*IMG_DPI, dtype=np.float64)
    ims = np.linspace(IM_MIN, IM_MAX, IMG_H_INCH*IMG_DPI, dtype=np.float64)
    c_region=np.empty([ims.shape[0], res.shape[0]], dtype=np.complex128)
    init_complex_region(res, ims, c_region)

    print("start to render mandelbrot in {}x{} pixel, max depth:{}, dpi:{}".format(
                IMG_W_INCH*IMG_DPI, IMG_H_INCH*IMG_DPI, MAX_ITERATION, IMG_DPI))
    depth = np.empty([c_region.shape[0],c_region.shape[1]], dtype=np.float64)
    start_time = timer()
    mandelbrot_region(c_region, depth)
    end_time = timer()
    print(" ...rendered in {} second".format(end_time-start_time))

    norm = colors.PowerNorm(GAMMA)
    ax.imshow(depth,cmap='magma',origin='lower',norm=norm, interpolation=INTERPOLATION)


    xticks = np.arange(0, IMG_W_INCH*IMG_DPI+1, IMG_W_INCH*IMG_DPI/4)
    xticks_delta = (RE_MAX-RE_MIN)/4.0
    xticks_label = np.arange(RE_MIN, RE_MAX+xticks_delta, xticks_delta,dtype=np.float64 )
    plt.xticks(xticks,['{:.6f}'.format(xl) for xl in xticks_label],fontsize='xx-small')
    
    yticks = np.arange(0, IMG_H_INCH*IMG_DPI+1, IMG_H_INCH*IMG_DPI/4)
    yticks_delt = (IM_MAX-IM_MIN)/4.0
    yticks_label = np.arange(IM_MIN, IM_MAX+yticks_delt, yticks_delt,dtype=np.float64 )
    plt.yticks(yticks, ['{:.6f}'.format(yl) for yl in yticks_label],fontsize='xx-small')
    plt.show()

    # save file
    fig.savefig("mandelbrot.jpg", dpi=IMG_DPI, bbox_inches='tight', pad_inches=0, frameon='false')

if __name__ == '__main__':
    main()
