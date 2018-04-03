
from lightkurve import KeplerTargetPixelFile
import numpy as np
import pandas as pd

tpf_unofficial = 'ktwo249622103-unofficial-tpf.fits'

tpf = KeplerTargetPixelFile(tpf_unofficial, quality_bitmask='hardest')
lc = tpf.to_lightcurve(aperture_mask='all');
lc = lc.remove_nans().remove_outliers(sigma=10)

#df contains
#BJD - 2454833	Raw Flux	Corrected Flux	X-centroid	Y-centroid	arclength	Correction	Thrusters On

time = lc.time
flux = lc.flux

col = lc.centroid_col
col = col - np.mean(col)
row = lc.centroid_row
row = row - np.mean(row)

#pd.DataFrame(np.c_[])

def _get_eigen_vectors(centroid_col, centroid_row):
    '''get the eigenvalues and eigenvectors given centroid x, y positions'''
    centroids = np.array([centroid_col, centroid_row])
    eig_val, eig_vec = np.linalg.eigh(np.cov(centroids))
    return eig_val, eig_vec

def _rotate(eig_vec, centroid_col, centroid_row):
    '''rotate the centroids into their predominant linear axis'''
    centroids = np.array([centroid_col, centroid_row])
    return np.dot(eig_vec, centroids)

eig_val, eig_vec = _get_eigen_vectors(col, row)
v1, v2 = eig_vec

platescale = 4.0 # The Kepler plate scale; has units of arcseconds / pixel

rot_colp, rot_rowp = _rotate(eig_vec, col, row) #units in pixels

## Calculate arclength
z = np.polyfit(rot_rowp, rot_colp, 5)
p5 = np.poly1d(z)
p5_deriv = p5.deriv()

x0_prime = np.min(rot_rowp)
xmax_prime = np.max(rot_rowp)
x_dense = np.linspace(x0_prime, xmax_prime, 2000)

@np.vectorize
def arclength(x):
    '''Input x1_prime, get out arclength'''
    gi = x_dense <x
    s_integrand = np.sqrt(1 + p5_deriv(x_dense[gi]) ** 2)
    s = np.trapz(s_integrand, x=x_dense[gi])
    return s

al = arclength(rot_rowp)*4.0


#apply high-pass filter using BSplines with N-day breakpoints
from scipy.interpolate import BSpline
from scipy import interpolate

N = 1.5
interior_knots = np.arange(time[0]+, time[0]+6, N)
t,c,k = interpolate.splrep(time, flux, s=0, task=-1, t=interior_knots)
bspl = BSpline(t,c,k)


flux = flux/bspl(time)

#remove data during thruster firing
good =
al = arclength(rot_rowp[good]) * platescale
