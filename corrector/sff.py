from scipy.interpolate import BSpline
from scipy import interpolate
import numpy as np

def rescale_centroids(col,row):
    col = col - np.mean(col)
    row = row - np.mean(row)
    return col, row

def get_eigen_vectors(centroid_col, centroid_row):
    '''get the eigenvalues and eigenvectors given centroid x, y positions'''
    centroids = np.array([centroid_col, centroid_row])
    eig_val, eig_vec = np.linalg.eigh(np.cov(centroids))
    return eig_val, eig_vec

def rotate(eig_vec, centroid_col, centroid_row):
    """Rotate the coordinate frame of the (col, row) centroids to a new (x,y)
    frame in which the dominant motion of the spacecraft is aligned with
    the x axis.  This makes it easier to fit a characteristic polynomial
    that describes the motion."""
    centroids = np.array([centroid_col, centroid_row])
    return np.dot(eig_vec, centroids)

def fit_curve(rot_rowp, rot_colp, order):
    z = np.polyfit(rot_rowp, rot_colp, order)
    p = np.poly1d(z)
    p_deriv = p.deriv()
    return p,p_deriv

def arclength(x_prime, x_dense, p_deriv):
    """
    Compute the arclength of the polynomial used to fit the centroid
    measurements.

    Parameters
    ----------
    x_prime : float
        Upper limit of the integration domain.
    x_dense : ndarray
        Domain at which the arclength integrand is defined.

    Returns
    -------
    arclength : float
        Result of the arclength integral from x[0] to x1.
    """

    s = []
    for i in x_prime:
        gi = x_dense < i
        s_integrand = np.sqrt(1 + p_deriv(x_dense[gi]) ** 2)
        s.append(np.trapz(s_integrand, x=x_dense[gi]))
    return np.array(s)

def find_thruster_events(time,data,Xc,Yc):
    '''
    Find events when the spacecruft thruster are fired.
    Usually no useful data points are gathered when this happens
    '''

    diff_centroid = np.diff(Xc)**2 + np.diff(Yc)**2

    thruster_mask = diff_centroid < (1.5*np.mean(diff_centroid) + 0.*np.std(diff_centroid))

    # this little trick helps us remove 2 data points each time instead of just 1
    thruster_mask1 = np.insert(thruster_mask,0, False)
    thruster_mask2 = np.append(thruster_mask,False)
    thruster_mask = thruster_mask1*thruster_mask2

    time_thruster = time[ thruster_mask ]
    diff_centroid_thruster = diff_centroid[ thruster_mask[1:] ]

#     Xc_clipped = Xc[:][thruster_mask]
#     Yc_clipped = Yc[:][thruster_mask]
#     time_clipped = time[:][thruster_mask]
#     data_clipped = data[:][thruster_mask]

    return ~thruster_mask

from matplotlib.colors import LogNorm
from scipy.ndimage import measurements
import os

cmap='viridis'

def make_aperture_outline(frame, no_combined_images=1, threshold=0.5):
    ## this is a little module that defines so called outlines to be used for plotting apertures

    thres_val = no_combined_images * threshold
    mapimg = (frame > thres_val)
    ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])
    hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan,np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))


    segments = np.array(l)

    x0 = -0.5
    x1 = frame.shape[1]+x0
    y0 = -0.5
    y1 = frame.shape[0]+y0

    #   now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    segments[:,0] = x0 + (x1-x0) * segments[:,0] / mapimg.shape[1]
    segments[:,1] = y0 + (y1-y0) * segments[:,1] / mapimg.shape[0]

    return segments

def find_aperture(dates,fluxes,starname='',kepmag='na',cutoff_limit=1.,showfig=None):
    #
    # This definition reads a 2D array of fluxes (over time) and creates an aperture mask which can later be used to select those pixels for inclusion in light curve
    #

    # first sum all the flux over the different times, this assumes limited movement throughout the time series
    flux = np.nansum(fluxes,axis=0)

    # define which cutoff flux to use for including pixel in mask
    cutoff = cutoff_limit*np.median(flux) # perhaps a more elaborate way to define this could be found in the future but this seems to work pretty well.

    # define the aperture based on cutoff and make it into array of 1 and 0
    aperture =  np.array([flux > cutoff]) #scipy.zeros((np.shape(flux)[0],np.shape(flux)[1]), int)
    aperture = np.array(1*aperture)
    #print aperture
    outline_all = make_aperture_outline(aperture[0]) # an outline (ONLY for figure) of what we are including if we would make no breakups

    # this cool little trick allows us to measure distinct blocks of apertures, and only select the biggest one
    lw, num = measurements.label(aperture) # this numbers the different apertures distinctly
    area = measurements.sum(aperture, lw, index=np.arange(lw.max() + 1)) # this measures the size of the apertures
    aperture = area[lw].astype(int) # this replaces the 1s by the size of the aperture
    aperture = (aperture >= np.max(aperture))*1 # remake into 0s and 1s but only keep the largest aperture

    outline = make_aperture_outline(aperture[0]) # a new outline (ONLY for figure)

    if showfig: # make aperture figure
        pl.figure('Aperture_' + str(starname))
        pl.imshow(flux,norm=LogNorm(),interpolation="none",cmap=cmap)
        pl.plot(outline_all[:, 0], outline_all[:, 1],color='green', zorder=10, lw=2.5)
        pl.plot(outline[:, 0], outline[:, 1],color='red', zorder=10, lw=2.5)#,label=str(kepmag))
        pl.colorbar(orientation='vertical')
        pl.xlabel('X',fontsize=15)
        pl.ylabel('Y',fontsize=15)
        #pl.legend()
        pl.tight_layout()
        pl.show()
    return np.array(aperture[0],dtype=bool)

import warnings
def get_centroids(flux, column, row, aperture_mask):
    """Returns centroids based on sample moments.

    Parameters
    ----------
    aperture_mask : array-like
        A boolean array describing the aperture such that `False` means
        that the pixel will be masked out.

    Returns
    -------
    col_centr, row_centr : tuple
        Arrays containing centroids for column and row at each cadence
    """
    yy, xx = np.indices(flux.shape[1:]) + 0.5
    yy = row + yy
    xx = column + xx
    total_flux = np.nansum(flux[:, aperture_mask], axis=1)
    with warnings.catch_warnings():
        # RuntimeWarnings may occur below if total_flux contains zeros
        warnings.simplefilter("ignore", RuntimeWarning)
        col_centr = np.nansum(xx * aperture_mask * flux, axis=(1, 2)) / total_flux
        row_centr = np.nansum(yy * aperture_mask * flux, axis=(1, 2)) / total_flux

    return col_centr, row_centr
