# Python code to compare starlight pol angles with radio pol angles
# Radio pol angle data comes from pas.py code via the .npz file
#
# 07-Jul-2020  C. Dickinson  Separated starlight code from pas.py into this separate code
#
# ----------------------------------------------------------------------
# modules
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import h5py
import os
from matplotlib.backends.backend_pdf import PdfPages
from clives_functions import *

# -----------------------------
# main parameters
nside_requested = 16  # Nside for main analyses
dir_string = 'v33b'  # string for main data file created by pp_polmaps.py (must exist)
sn_cutoff = 4  # Min S/N for most analyses and plots
sn_cutoff2 = 10  # higher S/N cut for some analyses
min_glat = 30.  # ***20 or 30 deg is a good value  (use 0 for plotting full sky)
min_PAgal_err = 1  # minimum error (deg) to consider for optical starlight polarization
max_PAgal_err = 10  # max error to consider useful
PAgal_err_cut = 2  # cut-off for error when closely comparing PAs
PAgal_err_default = 5.  # default uncertainty to assign when no error is available

# setup colours/parameters for plots
mycmap = get_cmap(colormap='erdc_iceFire')  # get cyclic colormap for colorblind (see Luke Jew's emails)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'  # set maths mode \mathrm to serif font

# -----------------------------
# First read in radio data
indata = np.load('pasdata_' + dir_string + '_h{0:.0f}.npz'.format(nside_requested))

# -----------------------------
# Check nside is what we want
nside = indata["nside"]
if (nside_requested != nside):
    print('Nside not the same as requested - please check!!!')
    quit()

# put data into arrays
map_freqs = indata["map_freqs"]
map_names = indata["map_names"]
maps = indata["maps"]
varmaps = indata["varmaps"]
pmaps = indata["pmaps"]
pmaps_err = indata["pmaps_err"]
pmaps_debiased = indata["pmaps_debiased"]
pmaps_sn = indata["pmaps_sn"]
pas = indata["pas"]
pas_err = indata["pas_err"]
pas_rotated = indata["pas_rotated"]
phi0_map = indata["phi0_map"]
phi0_err_map = indata["phi0_err_map"]
comb_index = indata["comb_index"]
cbass_index = indata["cbass_index"]
planck353_index = indata["planck353_index"]

# get useful information
nmaps = np.size(map_freqs)
npix = hp.nside2npix(nside)
ip = np.arange(npix)
glon, glat = hp.pix2ang(nside, ip, nest=False, lonlat='true')
glat_abs = np.abs(glat)
wavelengths = 2.997e8 / (map_freqs * 1e9)
wavelengths2 = wavelengths ** 2

# RA/Dec
# import astropy.units as u
# from astropy.coordinates import SkyCoord
# gc = SkyCoord(l=glon*u.degree, b=glat*u.degree, frame='galactic')
# radec = gc.fk5
# ra = radec.ra.deg
# dec = radec.dec.deg

# set plot directory
plotdir = 'plots_' + dir_string + '_h{0:d}/'.format(nside)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

# -----------------------------
# Read in optical starlight data from Gina (email 31 May 2019) - only works for Python 2.X
# import pickle

# fop = open('Starpol_catalog_berdyugin_heiles.pkl','r')
# data = pickle.load(fop)
# print data.keys()
# star_PAgal = data['PAgal'] # 1 d array, with each element being the measurement of a single star
# star_PAgal_err = data['e_EVPA']  # assuming error is same in C/G coordinate frames!
# star_glon  = data['GLON']
# star_glat  = data['GLAT']
# star_ra    = data['RA']
# star_dec   = data['Dec']
# star_dist  = data['Dist']

# instead read in numpy arrays (Gina email)
indir = 'berdyugin_heiles/'
star_PAgal = np.load(indir + 'PAgal.npy')
star_PAgal_err = np.load(indir + 'e_EVPA.npy')
star_glon = np.load(indir + 'GLON.npy')
star_glat = np.load(indir + 'GLAT.npy')
star_ra = np.load(indir + 'ra.npy')
star_dec = np.load(indir + 'Dec.npy')
star_dist = np.load(indir + 'Dist.npy')
star_p = np.load(indir + 'p.npy')
star_p_err = np.load(indir + 'e_p.npy')
nstar = np.size(star_PAgal)

# Rotate star angles by 90 deg to make them same as the radio
goodvals = np.isfinite(star_PAgal)
star_PAgal[goodvals] = fix_npi_ambiguity_deg(star_PAgal[goodvals] - 90.)  # this fixes pi ambiguity

# for NaN/-ve/0 PA errors give them a nominal 5 deg error for now (but not if star_PAgal = nan)
star_PAgal_err[(np.isnan(star_PAgal_err)) & (np.isfinite(star_PAgal))] = PAgal_err_default
goodvals = np.isfinite(star_PAgal_err)
star_PAgal_err[~goodvals] = 999  # set to large number (for converting to weight)
star_PAgal_err[(star_PAgal_err < 0.01)] = PAgal_err_default

# set a minimum value for the starlight errors
star_PAgal_err[star_PAgal_err < min_PAgal_err] = min_PAgal_err

# set bad (Nan) values to 0 so not to mess up the weighted average later
badvals = np.isnan(star_PAgal)
star_PAgal[badvals] = 0

# convert star PAgal errors to a weight and set NaNs to 0
star_PAgal_weight = 1 / star_PAgal_err ** 2
star_PAgal_weight[badvals] = 0  # set previously bad (Nan) values to 0 weight
star_PAgal_weight[np.isnan(star_PAgal_weight) | (star_PAgal_err > 45) | np.isnan(star_PAgal)] = 0.
star_PAgal_err[~goodvals] = 0.

# For -999 distances set to NaN
star_dist[np.isnan(star_dist)] = -999
# star_dist[star_dist < 0] = np.nan  # leave for now as it messes up comparisons later

# -------------------------------------------------------------------------
# bin the starlight values into a map for fun
map_star_Q = np.zeros(npix)
map_star_U = np.zeros(npix)
map_PAgal_weight = np.zeros(npix)
map_PAgal = np.zeros(npix)
map_nstar = np.zeros(npix)

# convert to Q/U
star_Q = np.cos(2 * star_PAgal * np.pi / 180)
star_U = np.sin(2 * star_PAgal * np.pi / 180)

# get Healpix pixel ID for each star
star_ipix = hp.ang2pix(nside, star_glon, star_glat, lonlat=True)

# add them into each Healpix pixel, weighting by the uncertainty
for i in np.arange(nstar):
    map_star_Q[star_ipix[i]] += star_Q[i] * star_PAgal_weight[i]
    map_star_U[star_ipix[i]] += star_U[i] * star_PAgal_weight[i]
    map_PAgal_weight[star_ipix[i]] += star_PAgal_weight[i]
    if (star_PAgal_weight[i] > 0):  # only add one if adds some weight
        map_nstar[star_ipix[i]] += 1

    # if (star_ipix[i] == 93):
    #    print(star_PAgal[i], star_PAgal_err[i])

# normalise by sum of weights
goodpix = map_PAgal_weight > 0.00
badpix = map_PAgal_weight <= 0.00
map_star_Q[goodpix] = map_star_Q[goodpix] / map_PAgal_weight[goodpix]
map_star_U[goodpix] = map_star_U[goodpix] / map_PAgal_weight[goodpix]

# get averaged pol angle
map_PAgal[goodpix] = qu2pa(map_star_Q[goodpix], map_star_U[goodpix], deg=True)
map_PAgal[goodpix] = fix_npi_ambiguity_deg(map_PAgal[goodpix])  # fix npi ambiguity

# get weighted error
# (*this will be an under-estimate as it assumes stars at same distance)
map_PAgal_err = np.zeros(npix) + hp.UNSEEN
map_PAgal_err[goodpix] = 1 / np.sqrt(map_PAgal_weight[goodpix])
map_PAgal_err[map_PAgal_err < min_PAgal_err] = min_PAgal_err  # set minimum error value

# set bad (no data) pixels back to Healpix badvalue
map_star_Q[badpix] = hp.UNSEEN
map_star_U[badpix] = hp.UNSEEN
map_PAgal[badpix] = hp.UNSEEN
map_nstar[badpix] = hp.UNSEEN
map_PAgal_err[badpix] = hp.UNSEEN

# including NaNs which seem to come from the original dataset
map_PAgal[np.isnan(map_PAgal)] = hp.UNSEEN
map_PAgal_err[np.isnan(map_PAgal)] = hp.UNSEEN

# get difference between each *individual* star angle and CBASS angle for that pixel
stars_cbass_diff = star_PAgal - pas[cbass_index, star_ipix]
stars_cbass_diff[np.isnan(stars_cbass_diff)] = -hp.UNSEEN
stars_cbass_diff[(stars_cbass_diff > (1e20))] = np.nan
goodvals = np.isfinite(stars_cbass_diff)
stars_cbass_diff[goodvals] = fix_npi_ambiguity_deg(stars_cbass_diff[goodvals])  # fixes pi ambiguity
stars_cbass_diff_err = np.sqrt(star_PAgal_err ** 2 + (pas_err[cbass_index, star_ipix]) ** 2)

# also for phi0
stars_phi0_diff = star_PAgal - phi0_map[star_ipix]
stars_phi0_diff[np.isnan(stars_phi0_diff)] = -hp.UNSEEN
stars_phi0_diff[(stars_phi0_diff > (1e20))] = np.nan
goodvals = np.isfinite(stars_phi0_diff)
stars_phi0_diff[goodvals] = fix_npi_ambiguity_deg(stars_phi0_diff[goodvals])  # fixes pi ambiguity
stars_phi0_diff_err = np.sqrt(star_PAgal_err ** 2 + (phi0_err_map[star_ipix]) ** 2)

# also for WMAP/Planck combination
wp_map = pas[comb_index, :]
wp_err_map = pas_err[comb_index, :]
stars_wp_diff = star_PAgal - wp_map[star_ipix]
stars_wp_diff[np.isnan(stars_wp_diff)] = -hp.UNSEEN
stars_wp_diff[(stars_wp_diff > (1e20))] = np.nan
goodvals = np.isfinite(stars_wp_diff)
stars_wp_diff[goodvals] = fix_npi_ambiguity_deg(stars_wp_diff[goodvals])  # fixes pi ambiguity
stars_wp_diff_err = np.sqrt(star_PAgal_err ** 2 + (wp_err_map[star_ipix]) ** 2)

# also for Planck 353
planck353_map = pas[planck353_index, :]
planck353_err_map = pas_err[planck353_index, :]
stars_planck353_diff = star_PAgal - planck353_map[star_ipix]
stars_planck353_diff[np.isnan(stars_planck353_diff)] = -hp.UNSEEN
stars_planck353_diff[(stars_planck353_diff > (1e20))] = np.nan
goodvals = np.isfinite(stars_planck353_diff)
stars_planck353_diff[goodvals] = fix_npi_ambiguity_deg(stars_planck353_diff[goodvals])  # fixes pi ambiguity
stars_planck353_diff_err = np.sqrt(star_PAgal_err ** 2 + (planck353_err_map[star_ipix]) ** 2)

# get difference between map of star angle and CBASS radio angle
star_cbass_diff = np.zeros(npix) + hp.UNSEEN
star_cbass_diff_err = np.zeros(npix) + hp.UNSEEN
goodpix = (map_PAgal_weight[:] > 0) & (pmaps_sn[cbass_index, :] > sn_cutoff)
star_cbass_diff[goodpix] = map_PAgal[goodpix] - pas[cbass_index, goodpix]
star_cbass_diff[goodpix] = fix_npi_ambiguity_deg(star_cbass_diff[goodpix])  # fixes pi ambiguity
star_cbass_diff_err[goodpix] = np.sqrt(map_PAgal_err[goodpix] ** 2 + (pas_err[cbass_index, goodpix]) ** 2)

# also get absolute differnece between stars and CBASS angle
star_cbass_diff_abs = np.zeros(npix) + hp.UNSEEN
star_cbass_diff_abs[goodpix] = np.abs(star_cbass_diff[goodpix])

# also get difference between phi0 model and map of stars
star_phi0_diff = np.zeros(npix) + hp.UNSEEN
star_phi0_diff_err = np.zeros(npix) + hp.UNSEEN
goodpix = (map_PAgal > -1e10) & (phi0_map > -1e10)
star_phi0_diff[goodpix] = map_PAgal[goodpix] - phi0_map[goodpix]
star_phi0_diff[goodpix] = fix_npi_ambiguity_deg(star_phi0_diff[goodpix])  # fixes pi ambiguity
star_phi0_diff_err[goodpix] = np.sqrt(map_PAgal_err[goodpix] ** 2 + (phi0_err_map[goodpix]) ** 2)

# also get absolute difference
star_phi0_diff_abs = np.zeros(npix) + hp.UNSEEN
star_phi0_diff_abs[goodpix] = np.abs(star_phi0_diff[goodpix])

# also get difference between WMAP/combination and map of stars
star_wp_diff = np.zeros(npix) + hp.UNSEEN
star_wp_diff_err = np.zeros(npix) + hp.UNSEEN
goodpix = (map_PAgal > -1e10) & (wp_map > -1e10)
star_wp_diff[goodpix] = map_PAgal[goodpix] - wp_map[goodpix]
star_wp_diff[goodpix] = fix_npi_ambiguity_deg(star_wp_diff[goodpix])  # fixes pi ambiguity
star_wp_diff_err[goodpix] = np.sqrt(map_PAgal_err[goodpix] ** 2 + (wp_err_map[goodpix]) ** 2)

# also get absolute difference
star_wp_diff_abs = np.zeros(npix) + hp.UNSEEN
star_wp_diff_abs[goodpix] = np.abs(star_wp_diff[goodpix])

# also get difference between Planck 353 and map of stars
star_planck353_diff = np.zeros(npix) + hp.UNSEEN
star_planck353_diff_err = np.zeros(npix) + hp.UNSEEN
goodpix = (map_PAgal > -1e10) & (planck353_map > -1e10)
star_planck353_diff[goodpix] = map_PAgal[goodpix] - planck353_map[goodpix]
star_planck353_diff[goodpix] = fix_npi_ambiguity_deg(star_planck353_diff[goodpix])  # fixes pi ambiguity
star_planck353_diff_err[goodpix] = np.sqrt(map_PAgal_err[goodpix] ** 2 + (planck353_err_map[goodpix]) ** 2)

# also get absolute difference
star_planck353_diff_abs = np.zeros(npix) + hp.UNSEEN
star_planck353_diff_abs[goodpix] = np.abs(star_wp_diff[goodpix])

# plot map of PAgal
plt.rcdefaults()  # reset plotting environment
hp.mollview(map_PAgal, title='Weighted star PA from Berdyugin+Heiles catalog', cmap=plt.cm.hsv, unit='Degrees', min=-90,
            max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_PAgal.png', dpi=300)
plt.clf()
plt.close()

# also map of PAgal error
hp.mollview(map_PAgal_err, title='Weighted star PA error from Berdyugin+Heiles catalog', cmap=plt.cm.jet,
            unit='Degrees', min=min_PAgal_err, max=20)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_PAgal_err.png', dpi=300)
plt.clf()
plt.close()

# plot map of Number of stars / pixel
hp.mollview(map_nstar, title='Number of stars in each Nside={0:d} pixel'.format(nside), max=10, cmap=plt.cm.jet)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'map_nstar.png', dpi=300)
plt.clf()
plt.close()

# plot map of CBASS PA (uncorrected)
hp.mollview(pas[cbass_index, :], title='C-BASS PA (uncorrected)', cmap=plt.cm.hsv, unit='Degrees', min=-90, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'cbass_pa.png', dpi=300)
plt.clf()
plt.close()

# plot map of CBASS PA (corrected)
hp.mollview(pas_rotated[cbass_index, :], title='C-BASS PA (FR corrected)', cmap=plt.cm.hsv, unit='Degrees', min=-90,
            max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'cbass_pa_corrected.png', dpi=300)
plt.clf()
plt.close()

# plot of difference map between stars and CBASS PA (uncorrected)
hp.mollview(star_cbass_diff, title=r'$\Delta \phi = \phi_{\mathrm{CBASS}} - \phi_{*}$ (deg.)', cmap=mycmap,
            unit='Degrees', min=-90, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_cbass_diff.png', dpi=300)
plt.clf()
plt.close()

# plot of absolute difference map between stars and CBASS PA (uncorrected)
hp.mollview(star_cbass_diff_abs, title=r'$|\Delta \phi| = |\phi_{\mathrm{CBASS}} - \phi_{*}|$ (deg.)', cmap=plt.cm.jet,
            unit='Degrees', min=0, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_cbass_diff_abs.png', dpi=300)
plt.clf()
plt.close()

# plot of difference map between stars and phi0
hp.mollview(star_phi0_diff, title=r'$\Delta \phi = \phi_0 (model) - \phi_{*}$ (deg.)', cmap=mycmap, unit='Degrees',
            min=-90, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_phi0_diff.png', dpi=300)
plt.clf()
plt.close()

# plot of absolute difference map between stars and phi0
hp.mollview(star_phi0_diff_abs, title=r'$|\Delta \phi| = \phi_0 (model) - \phi_{*}$ (deg.)', cmap=plt.cm.jet,
            unit='Degrees', min=0, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_phi0_diff_abs.png', dpi=300)
plt.clf()
plt.close()

# plot of difference map between stars and WMAP/Planck combination
hp.mollview(star_wp_diff, title=r'$\Delta \phi = \phi_0 (WMAP+Planck) - \phi_{*}$ (deg.)', cmap=mycmap, unit='Degrees',
            min=-90, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_wp_diff.png', dpi=300)
plt.clf()
plt.close()

# plot of absolute difference map between stars and WMAP/Planck combination
hp.mollview(star_wp_diff_abs, title=r'$|\Delta \phi| = \phi_0 (WMAP+Planck) - \phi_{*}$ (deg.)', cmap=plt.cm.jet,
            unit='Degrees', min=0, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_wp_diff_abs.png', dpi=300)
plt.clf()
plt.close()

# plot of difference map between stars and Planck 353
hp.mollview(star_planck353_diff, title=r'$\Delta \phi = \phi_0 (Planck 353) - \phi_{*}$ (deg.)', cmap=mycmap, unit='Degrees',
            min=-90, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_planck353_diff.png', dpi=300)
plt.clf()
plt.close()

# plot of absolute difference map between stars and Planck 353
hp.mollview(star_planck353_diff_abs, title=r'$|\Delta \phi| = \phi_0 (Planck 353) - \phi_{*}$ (deg.)', cmap=plt.cm.jet,
            unit='Degrees', min=0, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_planck353_diff_abs.png', dpi=300)
plt.clf()
plt.close()

# plot a map of the starlight data positions
blank_map = np.zeros(npix)
# hp.mollview(None,title='Star positions from Berdyugin+Heiles catalog')
hp.mollview(blank_map, title='Star positions from Berdyugin+Heiles catalog')
hp.graticule(30, verbose=False)
hp.projscatter(star_glon, star_glat, lonlat=True, marker='.', s=5, color='blue')
plt.savefig(plotdir + 'star_positions.png', dpi=300)
plt.clf()
plt.close()

# plot map of Planck 353 GHz PA
hp.mollview(pas[planck353_index, :], title='Planck 353 GHz PA', cmap=plt.cm.hsv, unit='Degrees', min=-90, max=90)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'planck353_pa.png', dpi=300)
plt.clf()
plt.close()

# ------------------------------------
# now do the PA difference again, but do it in bins of distance
dist_bins_low = np.array([0, 100, 200, 300, 400, 500, 1000, 2000, 3000])
dist_bins_high = np.append(dist_bins_low[1:], 30000)
dist_bins = (dist_bins_low + dist_bins_high) / 2
nbins = np.size(dist_bins)

map_star_Q_bin = np.zeros((nbins, npix))
map_star_U_bin = np.zeros((nbins, npix))
map_PAgal_bin = np.zeros((nbins, npix))
map_PAgal_weight_bin = np.zeros((nbins, npix))
padiff_bin = np.zeros((nbins, npix)) + hp.UNSEEN
padiff_bin_planck353 = np.zeros((nbins, npix)) + hp.UNSEEN

# loop over each distance bin
for j in np.arange(nbins):

    # get stars in each bin
    goodstars = np.array(np.nonzero(
        (star_dist >= dist_bins_low[j]) & (star_dist < dist_bins_high[j]) & (star_PAgal_weight > 0))).flatten()
    ngoodstars = np.size(goodstars)

    # get Healpix pixel ID for each star in this bin
    star_ipix_bin = hp.ang2pix(nside, star_glon[goodstars], star_glat[goodstars], lonlat=True)

    for i in np.arange(ngoodstars):
        map_star_Q_bin[j, star_ipix_bin[i]] += star_Q[goodstars[i]] * star_PAgal_weight[goodstars[i]]
        map_star_U_bin[j, star_ipix_bin[i]] += star_U[goodstars[i]] * star_PAgal_weight[goodstars[i]]
        map_PAgal_weight_bin[j, star_ipix_bin[i]] += star_PAgal_weight[goodstars[i]]

    # normalise by sum of weights
    goodpix = map_PAgal_weight_bin[j, :] > 0.00
    badpix = map_PAgal_weight_bin[j, :] <= 0.00
    map_star_Q_bin[j, goodpix] = map_star_Q_bin[j, goodpix] / map_PAgal_weight_bin[j, goodpix]
    map_star_U_bin[j, goodpix] = map_star_U_bin[j, goodpix] / map_PAgal_weight_bin[j, goodpix]

    # get averaged pol angle
    map_PAgal_bin[j, goodpix] = qu2pa(map_star_Q_bin[j, goodpix], map_star_U_bin[j, goodpix], deg=True)

    # set bad (no data) pixels back to Healpix badvalue
    map_star_Q_bin[j, badpix] = hp.UNSEEN
    map_star_U_bin[j, badpix] = hp.UNSEEN
    map_PAgal_bin[j, badpix] = hp.UNSEEN

    # get difference between each star angle and radio angle for that pixel where radio data are ok
    # goodpix = np.array(np.nonzero((map_PAgal_weight_bin[j,:] > 0.0) & (pmaps_sn[cbass_index,:] > sn_cutoff))).flatten()
    goodpix = np.array(np.nonzero((map_PAgal_weight_bin[j, :] > 0.0) & (pmaps_sn[comb_index, :] > sn_cutoff))).flatten()
    padiff_bin[j, goodpix] = fix_npi_ambiguity_deg(map_PAgal_bin[j, goodpix] - pas[comb_index, goodpix])
    padiff_bin_planck353[j, goodpix] = fix_npi_ambiguity_deg(map_PAgal_bin[j, goodpix] - pas[planck353_index, goodpix])

# take the abs value accounting for missing pixels
padiff_bin_abs = padiff_bin
goodvals = padiff_bin_abs > -1e10
padiff_bin_abs[goodvals] = np.abs(padiff_bin_abs[goodvals])

# also for Planck 353 takes abs
padiff_bin_planck353_abs = padiff_bin_planck353
goodvals = padiff_bin_planck353_abs > -1e10
padiff_bin_planck353_abs[goodvals] = np.abs(padiff_bin_planck353_abs[goodvals])

# plot the PA differences in different distance bins
# setup plot
plt.rc('font', family='serif', size=14)
figure = plt.gcf()
figure.set_size_inches([22, 14])
# plt.tight_layout()
pp = PdfPages(plotdir + 'padiff_dist.pdf')
plotcounter = 0  # counter for plots on a page
nplots_x = 3
nplots_y = 3
nplots_page = nplots_x * nplots_y
plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95, wspace=0.05, hspace=0.2)

# plot each pa difference for each distance bin
for j in np.arange(nbins):
    plotcounter = plotcounter + 1
    plt.subplot(nplots_y, nplots_x, plotcounter)
    hp.mollview(padiff_bin_abs[j, :],
                title=r'$|\Delta \phi|$ between radio/starlight {0:4d}-{1:4d} pc'.format(dist_bins_low[j],
                                                                                         dist_bins_high[j]),
                cmap=plt.cm.jet, hold=True, min=0, max=90, unit='Degrees')
    hp.graticule(30, verbose=False)

# close file
pp.savefig()
plt.clf()
pp.close()

# also for Planck353, plot the PA differences in different distance bins
# setup plot
plt.rc('font', family='serif', size=14)
figure = plt.gcf()
figure.set_size_inches([22, 14])
# plt.tight_layout()
pp = PdfPages(plotdir + 'padiff_planck353_dist.pdf')
plotcounter = 0  # counter for plots on a page
nplots_x = 3
nplots_y = 3
nplots_page = nplots_x * nplots_y
plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95, wspace=0.05, hspace=0.2)

# plot each pa difference for each distance bin
for j in np.arange(nbins):
    plotcounter = plotcounter + 1
    plt.subplot(nplots_y, nplots_x, plotcounter)
    hp.mollview(padiff_bin_planck353_abs[j, :],
                title=r'$|\Delta \phi|$ between Planck 353/starlight {0:4d}-{1:4d} pc'.format(dist_bins_low[j],
                                                                                         dist_bins_high[j]),
                cmap=plt.cm.jet, hold=True, min=0, max=90, unit='Degrees')
    hp.graticule(30, verbose=False)

# close file
pp.savefig()
plt.clf()
pp.close()

# ------------------------------------
# get consistency (sigma) between stars and CBASS/phi0 etc.
# get the consistency between the two in terms of this sigma

# star - CBASS
goodpix = star_cbass_diff > -1e10
star_cbass_diff_sigma = np.zeros(npix) + hp.UNSEEN
star_cbass_diff_sigma[goodpix] = np.abs(star_cbass_diff[goodpix] / star_cbass_diff_err[goodpix])

# plot
plt.rc('font', family='serif', size=10)
hp.mollview(star_cbass_diff_sigma, title=r'Consistency between $\phi_{*}$ and $\phi_{\mathrm{CBASS}} (\sigma)$', min=0,
            max=5, unit=r'$\sigma$', cmap=plt.cm.jet)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_cbass_diff_signif.png'.format(nside), dpi=300)
plt.clf()
plt.close()

# Star - phi0
goodpix = star_phi0_diff > -1e10
star_phi0_diff_sigma = np.zeros(npix) + hp.UNSEEN
star_phi0_diff_sigma[goodpix] = np.abs(star_phi0_diff[goodpix] / star_phi0_diff_err[goodpix])

# plot
hp.mollview(star_phi0_diff_sigma, title=r'Consistency between $\phi_{*}$ and $\phi_{0} (\sigma)$', min=0, max=5,
            unit=r'$\sigma$', cmap=plt.cm.jet)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_phi0_diff_signif.png'.format(nside), dpi=300)
plt.clf()
plt.close()

# Star - WMAP/Planck angle
goodpix = star_wp_diff > -1e10
star_wp_diff_sigma = np.zeros(npix) + hp.UNSEEN
star_wp_diff_sigma[goodpix] = np.abs(star_wp_diff[goodpix] / star_wp_diff_err[goodpix])

# plot
hp.mollview(star_wp_diff_sigma, title=r'Consistency between $\phi_{*}$ and $\phi_{WP} (\sigma)$', min=0, max=5,
            unit=r'$\sigma$', cmap=plt.cm.jet)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_wp_diff_signif.png'.format(nside), dpi=300)
plt.clf()
plt.close()

# Star - Planck 353 angle
goodpix = star_planck353_diff > -1e10
star_planck353_diff_sigma = np.zeros(npix) + hp.UNSEEN
star_planck353_diff_sigma[goodpix] = np.abs(star_planck353_diff[goodpix] / star_planck353_diff_err[goodpix])

# plot
hp.mollview(star_planck353_diff_sigma, title=r'Consistency between $\phi_{*}$ and $\phi_{Planck353} (\sigma)$', min=0, max=5,
            unit=r'$\sigma$', cmap=plt.cm.jet)
hp.graticule(30, verbose=False)
plt.savefig(plotdir + 'star_planck353_diff_signif.png'.format(nside), dpi=300)
plt.clf()
plt.close()

# ------------------------------------
# plot histogram of the PA differences of stars and CBASS in each pixel (not for each star)
plt.rcdefaults()
plt.rc('font', family='serif', size=16)
figure.set_size_inches([11, 7])
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
goodvals = (star_cbass_diff > -1e10) & (pmaps_sn[cbass_index, :] > sn_cutoff) & (map_PAgal_err < 7.16)
plt.annotate('RMS={0:4.1f}'.format(np.std(star_cbass_diff[goodvals])), (0.02, 0.92), xycoords='axes fraction',
             label='_nolegend_')
plt.hist(star_cbass_diff[goodvals], bins=20, alpha=0.5, label=r'$S/N>{0:.0f}$'.format(sn_cutoff))
goodvals = (star_cbass_diff > -1e10) & (pmaps_sn[cbass_index, :] > sn_cutoff) & (map_PAgal_err < 7.16) & (
            np.abs(glat) > min_glat)
plt.hist(star_cbass_diff[goodvals], bins=20, alpha=0.5, label=r'$|b|>{0:.0f}$'.format(min_glat))
plt.annotate('RMS={0:4.1f}'.format(np.std(star_cbass_diff[goodvals])), (0.02, 0.84), xycoords='axes fraction',
             label='_nolegend_')
# plt.title(r'$\phi_{*}-\phi_{sync}$')
plt.ylabel('Number of pixels')
plt.xlabel(r'$\Delta \phi = \phi_{\mathrm{CBASS}}-\phi_{*}$ (deg.)')
plt.xlim((-90, 90))
plt.plot(([0, 0]), plt.ylim(), 'k--', alpha=0.5)
plt.legend()
plt.savefig(plotdir + 'star_cbass_diff_hist.png', dpi=300)
plt.clf()
plt.close()

# and again for phi0
plt.rcdefaults()
plt.rc('font', family='serif', size=16)
figure.set_size_inches([11, 7])
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
goodvals = (star_phi0_diff > -1e10)
plt.annotate('RMS={0:4.1f}'.format(np.std(star_phi0_diff[goodvals])), (0.02, 0.92), xycoords='axes fraction',
             label='_nolegend_')
plt.hist(star_phi0_diff[goodvals], bins=20, alpha=0.5, label=r'$S/N>{0:.0f}$'.format(sn_cutoff))
goodvals = (star_phi0_diff > -1e10) & (np.abs(glat) > min_glat)
plt.hist(star_phi0_diff[goodvals], bins=20, alpha=0.5, label=r'$|b|>{0:.0f}$'.format(min_glat))
plt.annotate('RMS={0:4.1f}'.format(np.std(star_phi0_diff[goodvals])), (0.02, 0.84), xycoords='axes fraction',
             label='_nolegend_')
# plt.title(r'$\phi_{*}-\phi_{sync}$')
plt.ylabel('Number of pixels')
plt.xlabel(r'$\Delta \phi = \phi_{0}-\phi_{*}$ (deg.)')
plt.xlim((-90, 90))
plt.plot(([0, 0]), plt.ylim(), 'k--', alpha=0.5)
plt.legend()
plt.savefig(plotdir + 'star_phi0_diff_hist.png', dpi=300)
plt.clf()
plt.close()

# and again for WP
plt.rcdefaults()
plt.rc('font', family='serif', size=16)
figure.set_size_inches([11, 7])
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
goodvals = (star_wp_diff > -1e10)
plt.annotate('RMS={0:4.1f}'.format(np.std(star_phi0_diff[goodvals])), (0.02, 0.92), xycoords='axes fraction',
             label='_nolegend_')
plt.hist(star_wp_diff[goodvals], bins=20, alpha=0.5, label=r'$S/N>{0:.0f}$'.format(sn_cutoff))
goodvals = (star_wp_diff > -1e10) & (np.abs(glat) > min_glat)
plt.hist(star_wp_diff[goodvals], bins=20, alpha=0.5, label=r'$|b|>{0:.0f}$'.format(min_glat))
plt.annotate('RMS={0:4.1f}'.format(np.std(star_wp_diff[goodvals])), (0.02, 0.84), xycoords='axes fraction',
             label='_nolegend_')
# plt.title(r'$\phi_{*}-\phi_{sync}$')
plt.ylabel('Number of pixels')
plt.xlabel(r'$\Delta \phi = \phi_{WP}-\phi_{*}$ (deg.)')
plt.xlim((-90, 90))
plt.plot(([0, 0]), plt.ylim(), 'k--', alpha=0.5)
plt.legend()
plt.savefig(plotdir + 'star_wp_diff_hist.png', dpi=300)
plt.clf()
plt.close()

# and again for Planck 353
plt.rcdefaults()
plt.rc('font', family='serif', size=16)
figure.set_size_inches([11, 7])
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
goodvals = (star_planck353_diff > -1e10)
plt.annotate('RMS={0:4.1f}'.format(np.std(star_planck353_diff[goodvals])), (0.02, 0.92), xycoords='axes fraction',
             label='_nolegend_')
plt.hist(star_planck353_diff[goodvals], bins=20, alpha=0.5, label=r'$S/N>{0:.0f}$'.format(sn_cutoff))
goodvals = (star_planck353_diff > -1e10) & (np.abs(glat) > min_glat)
plt.hist(star_planck353_diff[goodvals], bins=20, alpha=0.5, label=r'$|b|>{0:.0f}$'.format(min_glat))
plt.annotate('RMS={0:4.1f}'.format(np.std(star_planck353_diff[goodvals])), (0.02, 0.84), xycoords='axes fraction',
             label='_nolegend_')
plt.ylabel('Number of pixels')
plt.xlabel(r'$\Delta \phi = \phi_{Planck353}-\phi_{*}$ (deg.)')
plt.xlim((-90, 90))
plt.plot(([0, 0]), plt.ylim(), 'k--', alpha=0.5)
plt.legend()
plt.savefig(plotdir + 'star_planck353_diff_hist.png', dpi=300)
plt.clf()
plt.close()

# ------------------------------------
# make histogram of star distances
plt.rcdefaults()
figure.set_size_inches([10, 7])
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
plt.hist(star_dist, bins=1000)
plt.title('Stellar distances (pc)')
plt.xlabel('Distance (pc)')
plt.yscale('log')
plt.savefig(plotdir + 'dist_hist.png', dpi=300)
plt.clf()
plt.close()

# ------------------------------------
# plot specific regions for Gina (email 10-Jul-2019)
hp.gnomview(star_cbass_diff, rot=[30, 35, 0], reso=1.5, xsize=3500, max=50, unit='Degrees', cmap=plt.cm.jet)
hp.graticule(5, verbose=False)
plt.savefig(plotdir + 'star_cbass_diff_l30_b35.png', dpi=300)
plt.clf()
plt.close()

hp.gnomview(pas[cbass_index, :], rot=[30, 35, 0], reso=1.5, xsize=3500, cmap=plt.cm.hsv, unit='Degrees')
hp.graticule(5, verbose=False)
plt.savefig(plotdir + 'cbass_pa_l30_b35.png', dpi=300)
plt.clf()
plt.close()

# ------------------------------------
# compare angles/distance/pol fractions generally - similar plot to Sun, Landecker et al. (2015) Fig.6

# setup plot
plt.rc('font', family='serif', size=7)
figure = plt.gcf()
figure.set_size_inches([7, 11])
# plt.tight_layout()
pp = PdfPages(plotdir + 'angles_dist.pdf')
plotcounter = 0  # counter for plots on a page
nplots_x = 2
nplots_y = 5
nplots_page = nplots_x * nplots_y
plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.95, wspace=0.3, hspace=0.3)
alphaval = 0.3

# plot 1 p vs dist for all
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
goodvals = np.isfinite(star_dist)
plt.plot(star_dist[goodvals], star_p[goodvals], 'b+', alpha=alphaval)
plt.title('Pol fraction vs distance (all stars)')
plt.xlim(1, 2000)
plt.xscale('log')

# plot starlight angle for all
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
goodvals = (np.isfinite(star_dist))
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'b+', alpha=alphaval)
plt.xlim(1, 2000)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance (all stars)')

# plot p vs dist for |b| > glat_cut
# plotcounter = plotcounter + 1
# plt.subplot(nplots_y,nplots_x,plotcounter)
# goodvals = (np.isfinite(star_dist)) & (np.abs(star_glat) > glat_cut)
# plt.plot(star_dist[goodvals], star_p[goodvals], 'b+', alpha=alphaval)
# plt.title('Pol fraction vs distance (|b|>{0:.0f})'.format(glat_cut))
# plt.xlim(1,2000)
# plt.xscale('log')

# plot starlight angle - radio angle against distance |b| > glat_cut
# plotcounter = plotcounter + 1
# plt.subplot(nplots_y,nplots_x,plotcounter)
# goodvals = (np.isfinite(star_dist)) & (np.abs(star_glat) > 30)
# plt.plot(star_dist[goodvals], star_pa_diff[goodvals], 'b+', alpha=alphaval)
# plt.title(r'$\phi_{*}-\phi_{sync}$ vs distance ' + '(|b|>{0:0.0f})'.format(glat_cut))
# plt.xlim(1,2000)
# plt.xscale('log')
# plt.plot(plt.xlim(), np.array([0,0]), 'k--')  # zero line

# plot p vs dist for high S/N stars/pixels and not in plane where FR important
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
goodvals = (np.isfinite(star_dist)) & (np.abs(star_glat) > min_glat) & (star_PAgal_err < PAgal_err_cut)
plt.plot(star_dist[goodvals], star_p[goodvals], 'b+', alpha=alphaval)
plt.title('Pol fraction vs distance (high S/N & |b|>{0:0.0f})'.format(min_glat))
plt.xlim(1, 2000)
plt.xscale('log')

# plot starlight angle - radio angle against distance for high S/N stars/pixels
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'b+', alpha=alphaval)
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance ' + '(|b|>{0:0.0f})'.format(min_glat))
plt.xlim(1, 2000)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line

# plot p vs dist for NPS/Loop I
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
goodvals = (np.isfinite(star_dist)) & (star_glon > 28) & (star_glon < 40) & (star_glat > 35) & (star_glat < 60)
plt.plot(star_dist[goodvals], star_p[goodvals], 'b+', alpha=alphaval)
plt.title('Pol fraction vs distance (NPS/Loop I bright region)')
plt.xlim(1, 2000)
plt.xscale('log')

# plot starlight angle - radio angle against distance for bright bit of NPS/Loop I
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
goodvals = (np.isfinite(star_dist)) & (star_glon > 28) & (star_glon < 40) & (star_glat > 35) & (star_glat < 60)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'b+', alpha=alphaval)
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance (NPS/Loop I bright region)')
plt.xlim(1, 2000)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line

# plot p near above l=0
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
goodvals = (np.isfinite(star_dist)) & (star_glon > 340) & (star_glon < 360) & (star_glat > 30) & (star_glat < 80)
plt.plot(star_dist[goodvals], star_p[goodvals], 'b+', alpha=alphaval)
plt.title('Pol fraction vs distance (above l=0)')
plt.xlim(1, 2000)
plt.xscale('log')

# plot starlight angle - radio angle against distance above l=0
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'b+', alpha=alphaval)
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance (above l=0)')
plt.xlim(1, 2000)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line

# plot p near loop IV
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
goodvals = (np.isfinite(star_dist)) & (star_glon > 300) & (star_glon < 330) & (star_glat > 30) & (star_glat < 80)
plt.plot(star_dist[goodvals], star_p[goodvals], 'b+', alpha=alphaval)
plt.title('Pol fraction vs distance (near loop IV)')
plt.xlim(1, 2000)
plt.xscale('log')

# plot starlight angle - radio angle against distance near loop IV
plotcounter = plotcounter + 1
plt.subplot(nplots_y, nplots_x, plotcounter)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'b+', alpha=alphaval)
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance (near loop IV)')
plt.xlim(1, 2000)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line

# close file
pp.savefig()
plt.clf()
pp.close()

# -----------------------------------
# plot PA differences for NPS/Loop I, colour coded by glat to see if there is a trend
plt.rcdefaults()
figure.set_size_inches([10, 7])

# plot different ranges of glat by hand
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 5.) & (star_glon > 26) & (star_glon < 40) & (star_glat > 20) & (
            star_glat < 30)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'b+', alpha=alphaval + 0.3)
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 5.) & (star_glon > 26) & (star_glon < 40) & (star_glat > 30) & (
            star_glat < 40)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'r+', alpha=alphaval + 0.3)
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 5.) & (star_glon > 26) & (star_glon < 40) & (star_glat > 40) & (
            star_glat < 60)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'g+', alpha=alphaval + 0.3)
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 5.) & (star_glon > 26) & (star_glon < 40) & (star_glat > 60) & (
            star_glat < 80)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'c+', alpha=alphaval + 0.3)
plt.legend(['20-30', '30-40', '40-60', '60-80'])
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance (NPS/Loop I bright region)')
plt.xlim(10, 2000)
plt.ylim(-90, 90)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line
plt.xlabel('Distance (pc)')
plt.ylabel('PA difference (deg.)')
plt.savefig(plotdir + 'stars_phi0_dist_NPS.png', dpi=300)
plt.clf()
plt.close()

# -------------------------------------------------------------------------
# do the same but look at the top and far side of Loop I to see if it gives the same overall distance
plt.rcdefaults()
figure.set_size_inches([10, 7])

# plot different ranges of glat by hand
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 5.) & (star_glon > 280) & (star_glon < 300) & (
            star_glat > 40) & (star_glat < 70)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'b+', alpha=alphaval + 0.3)
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance (NPS/Loop I far)')
plt.xlim(10, 2000)
plt.ylim(-90, 90)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line
plt.xlabel('Distance (pc)')
plt.ylabel('PA difference (deg.)')
plt.savefig(plotdir + 'stars_phi0_dist_NPSfar.png', dpi=300)
plt.clf()
plt.close()

# -------------------------------------------------------------------------
# plot for IV arch (l,b)=(120,50)
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 90.) & (star_glon > 110) & (star_glon < 130) & (
            star_glat > 48) & (star_glat < 52)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'bo', alpha=alphaval + 0.3)
plt.errorbar(star_dist[goodvals], stars_phi0_diff[goodvals], yerr=stars_phi0_diff_err[goodvals], ls='none')
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance (IV arch: (l,b)=(120,+50))')
plt.xlim(1, 30000)
plt.ylim(-90, 90)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line
plt.xlabel('Distance (pc)')
plt.ylabel('PA difference (deg.)')
plt.savefig(plotdir + 'stars_phi0_dist_IVarch.png', dpi=300)
plt.clf()
plt.close()

# -------------------------------------------------------------------------
# plot for bottom of loop II (l=145-180,b=-60 to -75)
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 90.) & (star_glon > 145) & (star_glon < 180) & (
            star_glat > -75) & (star_glat < -60)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'bo', alpha=alphaval + 0.3)
plt.errorbar(star_dist[goodvals], stars_phi0_diff[goodvals], yerr=stars_phi0_diff_err[goodvals], ls='none')
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance (bottom loop II: (l,b)=(145-180,-75--60))')
plt.xlim(1, 30000)
plt.ylim(-90, 90)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line
plt.xlabel('Distance (pc)')
plt.ylabel('PA difference (deg.)')
plt.savefig(plotdir + 'stars_phi0_dist_loopIIbottom.png', dpi=300)
plt.clf()
plt.close()

# -------------------------------------------------------------------------
# plot PA differences for regions above the plane, colour coded by glat to see if there is a trend
plt.rcdefaults()
figure.set_size_inches([10, 7])

# plot different ranges of glat by hand
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 10.) & (star_glon > 340) & (star_glon < 360) & (
            star_glat > 30) & (star_glat < 40)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'b+', alpha=alphaval + 0.3)
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 10.) & (star_glon > 340) & (star_glon < 360) & (
            star_glat > 40) & (star_glat < 50)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'r+', alpha=alphaval + 0.3)
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 10.) & (star_glon > 340) & (star_glon < 360) & (
            star_glat > 50) & (star_glat < 60)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'g+', alpha=alphaval + 0.3)
goodvals = (np.isfinite(star_dist)) & (star_PAgal_err < 10.) & (star_glon > 340) & (star_glon < 360) & (
            star_glat > 60) & (star_glat < 80)
plt.plot(star_dist[goodvals], stars_phi0_diff[goodvals], 'c+', alpha=alphaval + 0.3)
plt.legend(['30-40', '40-50', '50-60', '60-80'])
plt.title(r'$\phi_{*}-\phi_{0}$ vs distance (above l=0)')
plt.xlim(10, 2000)
plt.ylim(-90, 90)
plt.xscale('log')
plt.plot(plt.xlim(), np.array([0, 0]), 'k--')  # zero line
plt.xlabel('Distance (pc)')
plt.ylabel('PA difference (deg.)')
plt.savefig(plotdir + 'stars_phi0_dist_Abovel0.png', dpi=300)
plt.clf()
plt.close()


# ========================================================================
# ========================================================================




