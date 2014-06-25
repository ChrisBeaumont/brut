#================================================================================
# Title: Angular correlation and auto-correlation function for bubbles and YSOs
# Authors: S. Kendrew, Max Planck Institute for Astronomy, Heidelberg, 2012
#================================================================================
# Code names: calc_corr.py
#
# Language: python
#
# Code tested under the following compilers/operating systems: Mac OSX 10.6.8, python 2.6.6
#
# Description of input data:
# bubCat: a numpy recarray describing the input bubble catalog. must contain the following fields:
#        - bubCat['lon']: galactic longitude in degrees. longitudes > 180d must be converted to their negative equivalents
#        - bubCat['lat']: galactic latitude in degrees.
#        - bubCat['reff']: effective radius of the bubbles (or other measure of size), in arcminutes
# ysoCat: a numpy recarray describing the YSO catalog. must contain the following fields:
#        - ysoCat['lon']: galactic longitude, as above
#        - ysoCat['lat']: galactic latitude, as above
# corrType: a string describing flavour of correlation function
#        - 'a' for auto-correlation; in this case the bubble catalog is not used
#        - 'x' for cross-correlation between the two catalogs [default]
# rSize: integer, size of the random catalog; this is a multiplier applied to the size of the input catalogs. [default = 10; recommended > 20]
# nbStrap: integer, number of bootstrap ietrations performed [default = 20; recommended > 50]
# binStep: float, size of the separation bins between the sources. For auto-correlations this should be given in arcminutes, for cross-correlations in units of bubble Reff [default = 0.2]
#
#
# Description of output data:
#
# theta: a vector of floats with the size bins (for auto-correlation in arcminutes, for cross-correlation in unites of bubble radii)
# corr: a vector of floats with correlation values in each bin. unitless.
# corErr: a vector of floats, 1-sigma uncertainty on the correlation values in each bin. unitless.
#
# System requirements: 32-bits, 4 GB RAM (estimate)
#
# Calls to external routines:
#
# The code uses the following python modules:
#        numpy (module random)
#        scipy (modules stats, optimize)
#        matplotlib (module pyplot)
#        math
#        itertools
#        pdb (optional, for debugging)
# These packages are all explicitly imported at the start of the code; must be preinstalled by the user.
#
# Additional comments:
#
# The code calculates cross- and auto-correlation functions for two generic input catalgos, as described above. It uses the Landy-Szalay
# correlation estimator, described in detail in Landy & Szalay (1993).
# As well as the main function calc_corr, the file contains a number of supporting routines:
#    - fitLat: performs a Gaussian fit to the latitude distributions of the catalogs
#    - fitReff: performs a log-normal fit to the effective radii distribution of bubCat
#    - genRandomYso: generates a random catalog of YSOs based on the properties of the input catalog ysoCat and the specified size, rSize
#    - genRandomBubs: generates a random catalog of bubbles based on the properties of the input catalog bubCat and the specified size, rSize
#    - genbStrap: generates random indices with replacement of one of the input catalogs for the bootstrapping operation
#    - genNcountsX: performs pair counts for the correlation calculation for either corrType 'a' or 'x'
#    - genDiagFig: generates diagnostic figures comparing data and random catalog distributions with lon, lat and reff - for sanity check.
#    - genBoxFig: generates diagnostic box-and-whisker plot of pair counts per bin over all bootstrap iteration - for sanity check.
#
# Each of these functions contains a short description of inputs and outputs with its definition.
#
# This function can easily be integrated into a python script by adding
#                import calc_corr
# to the script header. Individual functions can then be called using e.g.
#                x, y, yerr = calc_corr.calc_corr(cat1, cat2, corrType='x', rSize=100, nbStrap=100, binStep=0.1)
#
#================================================================================
#The AAS gives permission to anyone who wishes to use these subroutines to run their own calculations.
#Permission to republish or reuse these routines should be directed to permissions@aas.org.

#Note that the AAS does not take responsibility for the content of the source code. Potential users should
#be wary of applying the code to conditions that the code was not written to model and the accuracy of the
#code may be affected when compiled and executed on different systems.
#================================================================================
#
#
# Written by S. Kendrew, 2012, kendrew@mpia.de
#
# Changes:
# jun 2013: changed addressing of recarray from e.g. ['lon'] to ['lon'] (in line with python 2.7)
#
#########################################################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize
import itertools
# Debugger: useful but optional
import pdb

from sklearn.neighbors import KDTree


def constrained_random(size, proposal, constraint):
    """
    Generate samples from a random distribution, with constraints

    Parameters
    ----------
    size : int
        Number of samples to draw

    proposal : func
        Function which takes size as input,
        and returns that many random samples

    constraint : func
        Function which takes an array as input,
        and returns a True/False array indicating
        whether each point is allowed

    Returns
    -------
    size draws from the proposal function, all of which pass the
    constraint test
    """
    result = proposal(size)
    bad = ~constraint(result)
    while bad.any():
        result[bad] = proposal(bad.sum())
        bad = ~constraint(result)
    return result


def fast_histogram(tree, xy, bins):
    """
    Use a KD tree to quickly compute a histogram of distances
    to an x, y point

    Parameters
    ----------
    tree : sklearn.neighbors.KDTree instance
        Tree holding the data points

    xy : ndarray [1 row, 2 columns]
        The point to query

    bins : ndarray [n elements]
        The bin edges

    Returns
    -------
    counts : [n - 1 elements]
       Histogram counts
    """
    #nudge xy, since KDtree doesn't like exact matches
    eps = np.random.normal(0, 1e-7, xy.shape)
    xy = xy + eps

    xy, _ = np.broadcast_arrays(xy.reshape(1, 2), bins.reshape(-1, 1))
    counts = tree.query_radius(xy, bins, count_only=True)
    return counts[1:] - counts[:-1]


#==============================================================
def fitLat(inpCat):
    ###########################################################
    # Fits a gaussian to the galactic latitude distribution of the input cat
    # inpCat: numpy recarray that must have the field inpCat['lat']
    # returns:
    # latp1[1]: mean of the best-fit gaussian (float)
    # latp1[2]: sigma of the best-fit gaussian (float)
    ###########################################################
    # define a gaussian fitting function where
    # p[0] = amplitude
    # p[1] = mean
    # p[2] = sigma
    latbins = np.arange(-1.0, 1.1, 0.1)
    hlat = np.histogram(inpCat['lat'], bins = latbins)
    #output from np.histogram: hlat[0] is counts per bin, hlat[1] gives the L edges of the bins
    fitfunc = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2.0*p[2]**2))
    errfunc = lambda p, x, y: fitfunc(p,x)-y
    latp0 = [np.max(hlat[0]), 0., 0.5]
    # fit a gaussian to the correlation function
    latx = 0.05+hlat[1][:-1]
    latp1, latsuccess = optimize.leastsq(errfunc, latp0,args=(latx, hlat[0]))

    # compute the best fit function from the best fit parameters
    latfit = fitfunc(latp1, latx)
    err_latfit = errfunc(latp1, latx, hlat[0])
    return latp1[1], latp1[2]

#=============================================================
def fitReff(inpCat):
    ##########################################################
    # Fits a log-normal distribution to the distribution of effective radii of inpCat (bubble catalog)
    # inpCat: numpy recarray that must have the field inpCat['reff'] describing the objects' effective radii (arcmin)
    # returns:
    # reffp1[1]: mean of distribution (float)
    # reffp1[2]: sigma of distribution (float)
    #########################################################
    rhist = np.histogram(np.log(inpCat['reff']), bins=10)
    binStep = rhist[1][1]-rhist[1][0]
    fitx = (binStep/2.)+rhist[1][:-1]
    fitfunc = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2.0*p[2]**2))
    errfunc = lambda p, x, y: fitfunc(p,x)-y
    reffp0 = [np.max(rhist[0]), 0., 1.5]
    # fit a gaussian to the correlation function
    reffp1, reffsuc = optimize.leastsq(errfunc, reffp0,args=(fitx, rhist[0]))
    fit_label = 'mu = %.2f, sigma = %.2f' %(reffp1[1], reffp1[2])
    # compute the best fit function from the best fit parameters
    reffFit = fitfunc(reffp1, fitx)
    err_reffFit = errfunc(reffp1, fitx, rhist[0])
    return reffp1[1], reffp1[2]

#==============================================================
def genRandomYso(ysoCat, size, params):
    ###########################################################
    # Generates a random catalog of YSOs from the input catalog, the user-specified random size and the latitude fit parameters
    # Inputs:
    # inpCat: numpy recarray, must contain fields inpCat['lon'] (gal longitude), inpCat['lat'] (gal latitude)
    # size: int, desired number of sources in random catalog; calculated from the rSize parameter in the top-level function
    # params: a 2-element tuple of floats with the mu and sigma values of the best-fit gaussian latitude distribution, as returned by fitLat()
    # Returns:
    # randCat: numpy recarray with fields randCat['lon'] (float, gal longitudes) and randCat['lat'] (float, gal latitudes)
    # NOTE: this version of the code excludes |l| < 10. as this is not covered in the RMS survey. for other surveys, amend lines
    #           167-168.
    ###########################################################
    coordLims = [-65., 65., -1., 1.]
    randSize = size
    types = [('lon', '<f8'), ('lat', '<f8')]
    randCatArr = np.ndarray(randSize, dtype=types)

    def lon_p(size):
        return np.random.uniform(coordLims[0], coordLims[1], size)

    def lon_c(x):
        return np.abs(x) > 10.

    def lat_p(size):
        return np.random.normal(params[0], params[1], size)

    def lat_c(x):
        return np.abs(x) <= 1

    lon = constrained_random(size, lon_p, lon_c)
    lat = constrained_random(size, lat_p, lat_c)

    randCatArr['lon'] = lon
    randCatArr['lat'] = lat
    randCat = randCatArr.view(np.recarray)

    return randCat

#===============================================================
def genRandomBubs(bubCat, size, params, rparams):
    ###########################################################
    # Generates a random catalog of bubbles from the input catalog, the user-specified random size and the latitude and Reff fit parameters
    # Inputs:
    # bubCat: numpy recarray, must contain fields bubCat['lon'] (float, gal longitude), bubCat['lat'] (float, gal latitude), bubCat['reff'] (float, effective radius)
    # size: int, desired number of sources in random catalog; calculated from the rSize parameter in the top-level function
    # params: a 2-element tuple of floats with the mu and sigma values of the best-fit gaussian latitude distribution, as returned by fitLat()
    # rparams: as params for the best-fit log-normal distribution for reff, as returned by fitReff()
    # Returns:
    # randCat: numpy recarray with fields randCat['lon'] (float, gal longitudes), randCat['lat'] (float, gal latitudes), randCat['reff'] (float, effective radii in arcmin)
    # all sources in randCat cover the same coordinate range as bubCat and all sizes are within the minimum and maximum sizes in bubCat
    ###########################################################
    coordLims = [-65., 65., -1., 1.]
    rLims = [np.min(bubCat['reff']), np.max(bubCat['reff'])]
    types = [('lon', '<f8'), ('lat', '<f8'), ('reff', '<f8')]
    randCatArr = np.ndarray(size, dtype=types)

    reff_min = np.min(bubCat['reff'])
    reff_max = np.max(bubCat['reff'])

    #generate the randoms here but ensure that stays within the required coordinate range

    def lon_p(size):
        return np.random.uniform(coordLims[0], coordLims[1], size)

    def lon_c(x):
        return np.abs(x) > 10.

    def lat_p(size):
        return np.random.normal(params[0], params[1], size)

    def lat_c(x):
        return np.abs(x) <= 1

    def reff_p(size):
        return np.random.lognormal(mean=rparams[0], sigma=rparams[1],
                                   size=size)
    def reff_c(x):
        return (x >= reff_min) & (x <= reff_max)

    lon = constrained_random(size, lon_p, lon_c)
    lat = constrained_random(size, lat_p, lat_c)
    reff_r = constrained_random(size, reff_p, reff_c)

    randCatArr['lon'] = lon
    randCatArr['lat'] = lat
    randCatArr['reff'] = reff_r
    randCat = randCatArr.view(np.recarray)
    return randCat

#=================================================================
def genBstrap(inpCat):
    ##############################################################
    # Generates random indices with replacement from the inpCat, for bootstrapping
    # Input:
    # inpCat: numpy recarray
    # Returns:
    # integer vector of randomised indices with the same length as inpCat, with replacements
    ##############################################################
    nelem = np.size(inpCat)
    return np.random.randint(0, nelem, nelem)

#=================================================================
def genNcountsX(cat1, cat2, bins, ctype):
    ##############################################################
    # Calculates the pair counts between two catalogs, or within 1 catalog, as a function of separation
    # The two catalogs can be any combination of bubbles and YSOs, data or random, depending on the correlation type specified.
    # If one of the catalogs is a bubbles catalog, it must be passed in cat1
    # Inputs:
    # cat1: numpy recarray for first catalog. must contain fields cat1['lon'] and cat1['lat'], as above. if represents bubbles, must contain cat1['reff'] field in arcminutes
    # cat2: numpy recarray for 2nd catalog. must contain fields cat2['lon'] and cat2['lat'], as above.
    # bins: a vector of floats specifying the theta bins in which the pair counts will be calculated.
    # ctype: 'a' for auto-correlation, 'x' for cross-correlation. inherited from the top level parameter cType
    # Returns:
    # dbnBox: numpy vector of length np.size(bins-1), containing total pair counts in each bin theta, prior to normalisation. this is used for the diagnostic box plots only, not for further calculations
    # dbnTot: numpy vector of length np.size(bins-1), containing normalised pair counts in each bin, for calculation of w(theta)
    ##############################################################
    nsrc = np.size(cat1)
    dbn = np.zeros((len(bins)-1,nsrc))

    # cycle through the bubbles and calc vectorised distances to the YSOs. Then histogram them.

    l1, l2, b1, b2 = map(np.asarray, [cat1['lon'], cat2['lon'],
                                      cat1['lat'], cat2['lat']])
    lb1 = np.column_stack((l1, b1))
    lb2 = np.column_stack((l2, b2))
    tree = KDTree(lb2)

    if ctype == 'a':
        #for auto-correlation between homogeneous catalogues:
        for i in range(0,nsrc):
            #convert bins from arcmin to degrees
            dbn[:, i] = fast_histogram(tree, lb1[i], bins / 60.)

    elif ctype == 'x':
        #for cross-correlation between heterogeneous catalogues:
        reff = np.asarray(cat1['reff'])
        for i in range(0,nsrc):
            #convert bins from bubble radii to degrees
            dbn[:, i] = fast_histogram(tree, lb1[i], bins * reff[i] / 60)

    dbnTot = np.sum(dbn, axis = 1)
    dbnBox = dbnTot
    dbnTot = dbnTot/np.sum(dbnTot)
    return dbnBox, dbnTot

#================================================================
def genBoxFig(bins, dd, dr, rd, rr, name):
    ##############################################################
    # Generates a box plot showing the total range of pair counts over the bootstrap iterations
    # Inputs:
    # bins: numpy vector of floats with the theta bins
    # dd: numpy vector of floats, total number of pair counts over the bootstrap iterations in each bin, for the data-data pair counts
    # dr: as dd, for data-random pair counts
    # rd: as dd, for random-data pair counts
    # rr: as dd, for random-random pair counts
    # name: name of catalog, for additional labelling if desired
    # Returns:
    # The box plot
    ##############################################################
    boxFig = plt.figure(figsize=[6,8])
    ddBox = boxFig.add_subplot(211)
    plt.boxplot(dd, notch=0, sym='bx')
    ddBox.set_title('Data-Data')
    xlabs=bins.astype('|S3')
    ddBox.set_xticklabels(xlabs, fontsize='medium')
    drBox = boxFig.add_subplot(212)
    plt.boxplot(dr, notch=0, sym='bx')
    drBox.set_title('Data-Random')
    drBox.set_xticklabels(xlabs, fontsize='medium')
    ddBox.set_xlabel(r'bin ($\theta$)')
    ddBox.set_ylabel('pair counts')
    drBox.set_xlabel(r'bin ($\theta$)')
    drBox.set_ylabel('pair counts')
    boxFig.show()
#================================================================
def genDiagFig(bub, yso, bubR, ysoR):
    ##############################################################
    # Generates a composite figure showing the distributions with lon, lat and reff of the data and random catalogues.
    # For diagnostic purposes
    # Inputs:
    # bub: numpy recarray, bubble catalog (data); inherited from top-level input parameters
    # yso: numpy recarray, yso catalog (data); inherited from top-level input parameters
    # bubR: numpy recarray, bubble catalog (random); as returned by genRandomBubs()
    # ysoR: numpy recarray, yso catalog (random); as returned by genRandomYso()
    # Returns:
    # the figure
    #############################################################
    diagFig = plt.figure(figsize=(12,8))
    lonData = diagFig.add_axes([0.05, 0.7, 0.4, 0.2])
    latData = diagFig.add_axes([0.5, 0.7, 0.4, 0.2])
    lonRand = diagFig.add_axes([0.05, 0.4, 0.4, 0.2])
    latRand = diagFig.add_axes([0.5, 0.4, 0.4, 0.2])
    reffPlot = diagFig.add_axes([0.05, 0.1, 0.4, 0.2])
    # do the first plots on the data
    lonBub = lonData.hist(bub['lon'], bins = 50, histtype='step', label = 'bubbles', lw=2)
    lonYso = lonData.hist(yso['lon'], bins = lonBub[1], histtype='step', label = 'ysos', lw=2)
    lonLeg = lonData.legend(loc = 'best')
    #lonLeg.set_fontsize('small')
    lonData.set_title('Longitude distribution - Data')
    latBub = latData.hist(bub['lat'], bins = 10, histtype='step', label = 'bubbles', axes = latData, lw=2)
    latYso = latData.hist(yso['lat'], bins = latBub[1], histtype='step', label = 'ysos', axes = latData, lw=2)
    latLeg = latData.legend(loc = 'best')
    #latLeg.set_fontsize('small')
    latData.set_title('Latitude distribution - Data')
    reffData = reffPlot.hist(bub['reff'], bins = 20, histtype='step', label = 'data', axes = reffPlot, normed=True, lw=2)
    reffPlot.set_title('Reff distribution')
    # add the random catalogue data to the diagnostic plots
    lonBubR = lonRand.hist(bubR['lon'], bins = 50, histtype='step', label = 'bubbles', lw=2)
    lonYsoR = lonRand.hist(ysoR['lon'], bins = lonBubR[1], histtype='step', label = 'ysos', lw=2)
    lonLegR = lonRand.legend(loc = 'best')
    lonRand.set_title('Longitude distribution - Randoms')
    latBubR = latRand.hist(bubR['lat'], bins = 10, histtype='step', label = 'bubbles', axes = latData, lw=2)
    latYsoR = latRand.hist(ysoR['lat'], bins = latBubR[1], histtype='step', label = 'ysos', axes = latData, lw=2)
    latLegR = latRand.legend(loc = 'best')
    latRand.set_title('Latitude distribution - Random')
    reffRand = reffPlot.hist(bubR['reff'], bins = reffData[1], histtype='step', label = 'randoms', axes = reffPlot, normed=True, lw=2)
    reffLeg = reffPlot.legend(loc=0)
    diagFig.show()
#================================================================
def divSample(ysoCat, bubCat):
    #############################################################
    # Divide the YSOs up into "associated" and "control" samples according to distance to nearest bubble
    # Input:
    #   ysoCat, bubCat: as defined before
    # Returns:
    #   assoc: the YSOs that lie within 2 effective radii from a bubble
    #   assoc2: the YSOs that are specifically within 0.8-1.6 effective radii from a bubble (i.e. associated with the bubble rim)
    #   control: YSOs that lie further than 3 efefctive radii from the nearest bubble.
    #############################################################

    nyso=np.size(ysoCat)
    assoc = np.zeros(1, dtype=int)
    control = np.zeros(1, dtype=int)
    assoc2 = np.zeros(1, dtype=int)
    #control2 = np.zeros(1, dtype=int)
    for i in range(0,nyso):
        dist = np.sqrt((bubCat['lon'] - ysoCat['lon'][i])**2 + (bubCat['lat'] - ysoCat['lat'][i])**2)
        dist = dist*60.
        dist = dist/bubCat['reff']            #distance from the YSO to all bubbles, expressed as a function of their effective radius
        mindist = np.min(dist)
        #print str(mindist)+' Reff'
        if mindist <= 2.:
            assoc = np.append(assoc, i)
            if (mindist >= 0.8) & (mindist <= 1.6):
                assoc2 = np.append(assoc2, i)
            #print 'added to associated sample'
        if mindist > 3.:
            control = np.append(control, i)
            #print 'added to control sample'
    assoc = assoc[1:]                #remove the first element, which will be zero from how I've defined the array
    assoc2 = assoc2[1:]
    control = control[1:]
    print 'Bubble-associated sample contains %i YSOs; assoc2 contains %i sources. Control sample contains %i YSOs.' %(np.size(assoc), np.size(assoc2),np.size(control))
    return assoc, assoc2, control
#================================================================
def calcAcorr(dd, dr, rr, bins):
    ##############################################################
    # Calculates the auto-correlation function from pair counts returned by genNcountsX(), if corrType='a'
    # Inputs:
    # dd: numpy floats vector of normalised data-data pair counts in each theta bin
    # dr: numpy floats vector of normalised data-random pair counts in each theta bin
    # rr: numpy floats vector of normalised random-random pair counts in each theta bin
    # bins: numpy floats vector of theta bins (in arcminutes)
    # Returns:
    # w: numpy floats vector with auto-correlation values in each bin theta (see Landy & Szalay, 1993)
    ###############################################################
    w = np.zeros(len(bins))
    w = (dd-(2*dr)+rr)/rr
    return w

#================================================================
def calcXcorr(dd,dr,rd,rr, bins):
    ##############################################################
    # Calculates the cross-correlation function from pair counts returned by genNcountsX(), if corrType='x'
    # Inputs:
    # dd: numpy floats vector of normalised data-data pair counts in each theta bin
    # dr: numpy floats vector of normalised data-random pair counts in each theta bin
    # rd: numpy floats vector of normalised random-data pair counts in each theta bin
    # rr: numpy floats vector of normalised random-random pair counts in each theta bin
    # bins: numpy floats vector of theta bins (normalised to bubble radii)
    # Returns:
    # w: numpy floats vector with cross-correlation values in each bin theta (see Landy & Szalay, 1993)
    ###############################################################
    #calculate the angular correlation vector from the calculated number counts
    w = np.zeros(len(bins))
    w = (dd-dr-rd+rr)/rr
    return w

#==================================================================
# MAIN FUNCTION
#==================================================================
def calc_corr(bubCat, ysoCat, corrType='x', rSize=10, nbStrap=20, binStep=0.2):

    #find the best-fit gaussian params for the two input catalogues' latitude distribution:
    ysoLatmu, ysoLatsig = fitLat(ysoCat)
    ysoParams = [ysoLatmu, ysoLatsig]
    print 'fit parameters for ysos: mean = %.2f, sigma = %.2f' %(ysoLatmu, ysoLatsig)

    #if doing a cross-correlation with bubbles then we want the fits for the bubbles as well.
    if corrType == 'x':
        bubLatmu, bubLatsig = fitLat(bubCat)
        reffp1, reffp2 = fitReff(bubCat)
        bubParams = [bubLatmu, bubLatsig]
        reffParams = [reffp1, reffp2]
        print 'fit parameters for bubbles: mean = %.2f, sigma = %.2f' %(bubLatmu, bubLatsig)
        print 'fit parameters for bubble Reffs: mu = %.2f, sigma = %.2f' %(reffp1, reffp2)

    # Define random catalogue sizes:
    ysoRandSize = rSize*np.size(ysoCat)
    bubRandSize = rSize*np.size(bubCat)

    # create angular distance grid generate random cats. if doing a
    # cross-correlation then want both bubble, yso cats, if auto-correlation then only need
    # yso random cat.
    if corrType == 'x':
        theta = np.arange(0.0, 4., binStep)             #in Reff units
        ysoRand = genRandomYso(ysoCat, ysoRandSize, ysoParams)
        bubRand = genRandomBubs(bubCat, bubRandSize, bubParams, reffParams)
        genDiagFig(bubCat, ysoCat, bubRand, ysoRand)

    if corrType == 'a':
        theta = np.arange(0., 10., binStep)             #in arcminutes
        ysoRand = genRandomYso(ysoCat, ysoRandSize, ysoParams)
    nbins = np.size(theta)


    print 'Number of bootstrap iterations = %i' %(nbStrap)

    #create arrays for the pair counts
    corrAll = np.zeros((nbins,nbStrap))
    ddBoxdat = np.zeros((nbStrap, nbins-1))
    drBoxdat = np.zeros((nbStrap, nbins-1))
    rdBoxdat = np.zeros((nbStrap, nbins-1))
    rrBoxdat = np.zeros((nbStrap, nbins-1))



    # if corrType =='x' we want a cross-correlation between the bubbles and YSOs. As random-data and random-random don't use bootstrapped cats, can perform these outside of the bootstrap loop
    if corrType == 'x':
        # pair counts with the bubbles random catalogue are the same for each bootstrap iteration so can take these out fo the for loop
        print 'Random pair counts'
        rdBox, rdTotal = genNcountsX(bubRand, ysoCat, theta, corrType)
        rrBox, rrTotal = genNcountsX(bubRand, ysoRand, theta, corrType)
        for i in range(0,nbStrap):
            print 'Bootstrap iteration {0}' .format(i)
            bStrapInd = genBstrap(bubCat)
            bubCat_bStrap = bubCat[bStrapInd]
            ddBox, ddTotal = genNcountsX(bubCat_bStrap, ysoCat, theta, corrType)
            drBox, drTotal = genNcountsX(bubCat_bStrap, ysoRand, theta, corrType)
            corrTemp = calcXcorr(ddTotal, drTotal, rdTotal, rrTotal, theta)
            corrAll[:-1,i] = corrTemp
            ddBoxdat[i,:] = ddBox
            drBoxdat[i,:] = drBox
            rdBoxdat[i,:] = rdBox
            rrBoxdat[i,:] = rrBox

    # if corrType == 'a' I want the autocorrelation of the YSO catalogue only.
    elif corrType == 'a':
        rrBox, rrTotal = genNcountsX(ysoRand, ysoRand, theta, corrType)
        for i in range(0,nbStrap):
            bStrapInd = genBstrap(ysoCat)
            ysoCat_bStrap = ysoCat[bStrapInd]
            ddBox, ddTotal = genNcountsX(ysoCat_bStrap, ysoCat, theta, corrType)
            drBox, drTotal = genNcountsX(ysoCat_bStrap, ysoRand, theta, corrType)
            corrTemp = calcAcorr(ddTotal, drTotal, rrTotal, theta)
            corrAll[:-1,i] = corrTemp

    # Average the correlation functions over all the bootstrap iterations, and calculate the standard deviations:
    corr = np.mean(corrAll, axis=1)
    corrErr = np.std(corrAll, axis=1)

    # Plot the correlation function
    xcorrFig = plt.figure()
    plt.errorbar(theta, corr, yerr = corrErr, xerr = None, fmt = 'bo', figure=xcorrFig)
    title = '%s correlation function. Bootstrap iterations = %i, Random catalogue size = %i.' %(corrType, nbStrap, rSize)
    plt.suptitle(title)

    if corrType == 'x': plt.xlabel('bubble-YSO separation theta (theta/Reff)')
    plt.ylabel('w(theta)')

    if corrType == 'a':
        bub_med = stats.scoreatpercentile(bubCat['reff'], 50.)
        bubq1 = stats.scoreatpercentile(bubCat['reff'], 25.)
        bubq3 = stats.scoreatpercentile(bubCat['reff'], 75.)
        plt.axvline(x=bub_med, ymin=0, ymax=1, c='red', lw = 2., ls = '--')
        plt.axvline(x=bubq1, ymin=0, ymax=1, c='green', lw = 2., ls = '--')
        plt.axvline(x=bubq3, ymin=0, ymax=1, c='green', lw = 2., ls = '--')
        plt.xlabel('theta (arcmin)')

    if corrType == 'x':
        genBoxFig(theta,ddBoxdat, drBoxdat, rdBoxdat, rrBoxdat, bubCat)

    xcorrFig.show()

    return theta, corr, corrErr
