# modified from https://github.com/buenrostrolab/scPrinter/blob/main/scprinter/footprint.py
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
import warnings

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

# Doing the same thing as conv in R, but more generalizable
def rz_conv(a, n=2):
    if n == 0:
        return a
    # a can be shape of (batch, sample,...,  x) and x will be the dim to be conv on
    # pad first:
    shapes = np.array(a.shape)
    shapes[-1] = n
    a = np.concatenate([np.zeros(shapes), a, np.zeros(shapes)], axis=-1)
    ret = np.nancumsum(a, axis=-1)
    # ret[..., n * 2:] = ret[..., n * 2:] - ret[..., :-n * 2]
    # ret = ret[..., n * 2:]
    ret = ret[..., n * 2 :] - ret[..., : -n * 2]
    return ret

# Given a value matrix x, along the last dim,
# for every single position, calculate sum of values in center and flanking window
# Essentially we are doing a running-window sum for the center and flanking
def footprintWindowSum(
    x,  # A numerical or integer vector x
    footprintRadius,  # Radius of the footprint region
    flankRadius,  # Radius of the flanking region (not including the footprint region)
):
    """
    This function calculates the window sum of values in a given insertion vector 'x' for every position,
    considering a window of specified footprint and flanking radii.

    Parameters:
    x (numpy.ndarray): A numerical or integer vector of shape (n,) or (n, m), where n is the number of samples and m is the length of the vector.
    footprintRadius (int): The radius of the footprint region.
    flankRadius (int): The radius of the flanking region (not including the footprint region).

    Returns:
    dict: A dictionary containing the sum of values in the left flanking window, center window, and right flanking window.
    """
    backend_string = "numpy" if isinstance(x, np.ndarray) else "torch"

    halfFlankRadius = int(flankRadius / 2)
    width = x.shape[-1]

    # Calculate sum of x in the left flanking window
    shift = halfFlankRadius + footprintRadius
    # x can be shape of (x) or (sample, x)
    shapes = list(x.shape)
    shapes[-1] = shift

    leftShifted = np.concatenate([np.zeros(shapes), x], axis=-1)

    leftFlankSum = rz_conv(leftShifted, halfFlankRadius)[..., :width]

    # Calculate sum of x in the right flanking window
    rightShifted = np.concatenate([x, np.zeros(shapes)], axis=-1)
    rightFlankSum = rz_conv(rightShifted, halfFlankRadius)[..., shift:]

    centerSum = rz_conv(x, footprintRadius)

    return leftFlankSum, centerSum, rightFlankSum


def footprintScoring(
    Tn5Insertion,
    # Integer vector of raw observed Tn5 insertion counts at each single base pair
    Tn5Bias,  # Vector of predicted Tn5 bias. Should be the same length
    dispersionModel,
    # Background dispersion model for center-vs-(center + flank) ratio insertion ratio
    footprintRadius=10,  # Radius of the footprint region
    flankRadius=10,  # Radius of the flanking region (not including the footprint region)
    return_pval=True,
):
    """
    Calculates the footprint scoring for multiple samples based on the observed Tn5 insertion counts and
    predicted Tn5 bias for one single region and one scale (mode)

    Parameters:
    Tn5Insertion (numpy.ndarray): Integer vector of raw observed Tn5 insertion counts at each single base pair.
    Tn5Bias (numpy.ndarray): Vector of predicted Tn5 bias. Should be the same length as Tn5Insertion's last dim (genome).
    dispersionModel (dict): Background dispersion model for center-vs-(center + flank) ratio insertion ratio.
    footprintRadius (int, optional): Radius of the footprint region. Default is 10.
    flankRadius (int, optional): Radius of the flanking region (not including the footprint region). Default is 10.
    return_pval (bool, optional): Boolean indicating whether to return p-values. Default is True.

    Returns:
    numpy.ndarray: Numpy array of p-values or z-scores, depending on the value of return_pval.
    """

    modelWeights = dispersionModel["modelWeights"]

    # Get sum of predicted bias in left flanking, center, and right flanking windows
    biasleftFlankSum, biascenterSum, biasrightFlankSum = footprintWindowSum(
        Tn5Bias, footprintRadius, flankRadius
    )
    biasleftFlankSum, biascenterSum, biasrightFlankSum

    # Get sum of insertion counts in left flanking, center, and right flanking windows
    insertionleftFlankSum, insertioncenterSum, insertionrightFlankSum = footprintWindowSum(
        Tn5Insertion, footprintRadius, flankRadius
    )

    # print (insertionWindowSums['center'].shape, biasWindowSums['center'].shape)
    leftTotalInsertion = insertioncenterSum + insertionleftFlankSum
    rightTotalInsertion = insertioncenterSum + insertionrightFlankSum
    # Prepare input data (foreground features) for the dispersion model
    fgFeatures = np.stack(
        [
            np.array([biasleftFlankSum] * len(Tn5Insertion)),
            np.array([biasrightFlankSum] * len(Tn5Insertion)),
            np.array([biascenterSum] * len(Tn5Insertion)),
            np.log10(leftTotalInsertion),
            np.log10(rightTotalInsertion),
        ],
        axis=-1,
    )

    fgFeaturesScaled = (fgFeatures - dispersionModel["featureMean"]) / (
        dispersionModel["featureSD"]
    )
    # Given observed center bias, flank bias, and total insertion, use our model to estimate background
    # dispersion of and background mean of center-vs-(center + flank) ratio
    with torch.no_grad():
        predDispersion = predictDispersion_jit(
            torch.from_numpy(fgFeaturesScaled).float(), *modelWeights
        ).numpy()

    predDispersion = predDispersion * dispersionModel["targetSD"]
    predDispersion = predDispersion + dispersionModel["targetMean"]

    leftPredRatioMean = predDispersion[..., 0]
    leftPredRatioSD = predDispersion[..., 1]
    rightPredRatioMean = predDispersion[..., 2]
    rightPredRatioSD = predDispersion[..., 3]
    leftPredRatioSD[leftPredRatioSD < 0] = 0
    rightPredRatioSD[rightPredRatioSD < 0] = 0
    # Calculate foreground (observed) center-vs-(center + flank) ratio
    fgLeftRatio = insertioncenterSum / leftTotalInsertion
    fgRightRatio = insertioncenterSum / rightTotalInsertion

    if return_pval:
        # Compute p-value based on background mean and dispersion
        leftPval = scipy.stats.norm.cdf(fgLeftRatio, leftPredRatioMean, leftPredRatioSD)
        # This is to make it consistent with R pnorm
        leftPval[np.isnan(leftPval)] = 1
        rightPval = scipy.stats.norm.cdf(fgRightRatio, rightPredRatioMean, rightPredRatioSD)
        rightPval[np.isnan(rightPval)] = 1

        # Combine test results for left flank and right flank by taking the bigger pval
        p = np.maximum(leftPval, rightPval)

        # Mask positions with zero coverage on either flanking side
        p[(leftTotalInsertion < 1) | (rightTotalInsertion < 1)] = 1
        return p
    else:
        # return z-score
        leftZ = (fgLeftRatio - leftPredRatioMean) / leftPredRatioSD
        rightZ = (fgRightRatio - rightPredRatioMean) / rightPredRatioSD
        z = np.maximum(leftZ, rightZ)
        return z


# Calculate footprint score track for a single genomic region
def regionFootprintScore(
    regionATAC,
    Tn5Bias,
    dispersionModel,
    footprintRadius,
    flankRadius,
    extra_info,  # extra_info to be returned, so the parent process would know which child it is.
    smoothRadius=None,
    return_pval=True,
):
    """
    Calculates the pseudo-bulk-by-position footprint score matrix for a given genomic region and one scale.

    Parameters:
    regionATAC (numpy.ndarray): Integer vector of raw observed ATAC-seq counts at each single base pair.
    Tn5Bias (numpy.ndarray): Vector of predicted Tn5 bias. Should be the same length as regionATAC.
    dispersionModel (dict): Background dispersion model for center-vs-(center + flank) ratio insertion ratio.
    footprintRadius (int): Radius of the footprint region.
    flankRadius (int): Radius of the flanking region (not including the footprint region).
    extra_info (any): Extra information to be returned, so the parent process would know which child it is.
    smoothRadius (int, optional): Radius of the smoothing window. Default is None, which means it will be calculated as footprintRadius / 2.
    return_pval (bool, optional): Boolean indicating whether to return p-values. Default is True.

    Returns:
    numpy.ndarray: Pseudo-bulk-by-position footprint score matrix.
    any: Extra information.
    """
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")

        # Calculate the pseudo-bulk-by-position footprint pvalue matrix
        footprintPvalMatrix = footprintScoring(
            Tn5Insertion=regionATAC,
            Tn5Bias=Tn5Bias,
            dispersionModel=dispersionModel,
            footprintRadius=footprintRadius,
            flankRadius=flankRadius,
            return_pval=return_pval,
        )
        if return_pval:
            if smoothRadius is None:
                smoothRadius = int(footprintRadius / 2)
            footprintPvalMatrix[np.isnan(footprintPvalMatrix)] = 1  # Set NA values to be pvalue = 1
            # print (footprintPvalMatrix, np.sum(np.isnan(footprintPvalMatrix)), np.sum(np.isinf(footprintPvalMatrix)))
            pvalScoreMatrix = -np.log10(footprintPvalMatrix)
            pvalScoreMatrix[np.isnan(pvalScoreMatrix)] = 0
            pvalScoreMatrix[np.isinf(pvalScoreMatrix)] = 20
            if smoothRadius > 0:
                maximum_filter_size = [0] * len(pvalScoreMatrix.shape)
                maximum_filter_size[-1] = 2 * smoothRadius
                pvalScoreMatrix = maximum_filter(
                    pvalScoreMatrix, tuple(maximum_filter_size), origin=-1
                )
                # Changed to smoothRadius.
                pvalScoreMatrix = rz_conv(pvalScoreMatrix, smoothRadius) / (2 * smoothRadius)
            pvalScoreMatrix[np.isnan(pvalScoreMatrix)] = 0
            pvalScoreMatrix[np.isinf(pvalScoreMatrix)] = 20
        else:
            # pvalScoreMatrix = footprintPvalMatrix
            if smoothRadius is None:
                smoothRadius = int(footprintRadius / 2)
            # footprintPvalMatrix[np.isnan(footprintPvalMatrix)] = 1 # Set NA values to be pvalue = 1
            # print (footprintPvalMatrix, np.sum(np.isnan(footprintPvalMatrix)), np.sum(np.isinf(footprintPvalMatrix)))
            pvalScoreMatrix = footprintPvalMatrix
            # pvalScoreMatrix[np.isnan(pvalScoreMatrix)] = 0
            # pvalScoreMatrix[np.isinf(pvalScoreMatrix)] = 20
            if smoothRadius > 0:
                maximum_filter_size = [0] * len(pvalScoreMatrix.shape)
                maximum_filter_size[-1] = 2 * smoothRadius
                pvalScoreMatrix = maximum_filter(
                    pvalScoreMatrix, tuple(maximum_filter_size), origin=-1
                )
                # Changed to smoothRadius.
                pvalScoreMatrix = rz_conv(pvalScoreMatrix, smoothRadius) / (2 * smoothRadius)
            # pvalScoreMatrix[np.isnan(pvalScoreMatrix)] = 0
            # pvalScoreMatrix[np.isinf(pvalScoreMatrix)] = 20

    return pvalScoreMatrix, extra_info


def multiscale_footprints(
    region_ATAC,
    Tn5Bias,
    dispersionModels,
    modes=np.arange(2, 101),
    footprintRadius=None,  # Radius of the footprint region
    flankRadius=None,  # Radius of the flanking region (not including the footprint region)
    smoothRadius=5,
    extra_info=None,
    return_pval=True,
):
    """
    This function calculates pseudo-bulk-by-position footprint score matrices for multiple genomic regions
    using different footprint and flanking radii. This is used as a wrapper function to just get the multi-scale
    footprints without parallelization, mainly for visualization

    Parameters:
    region_ATAC (numpy.ndarray): Integer vector of raw observed ATAC-seq counts at each single base pair.
    Tn5Bias (numpy.ndarray): Vector of predicted Tn5 bias. Should be the same length as region_ATAC.
    dispersionModels (dict): Dictionary containing background dispersion models for different modes.
    modes (numpy.ndarray, optional): Array of modes (footprint radii) to be used. Default is np.arange(2, 101).
    footprintRadius (numpy.ndarray, optional): Array of footprint radii. If None, it will be set to modes. Default is None.
    flankRadius (numpy.ndarray, optional): Array of flanking radii. If None, it will be set to modes. Default is None.
    smoothRadius (int, optional): Radius of the smoothing window. Default is 5.
    extra_info (any, optional): Extra information to be returned, so the parent process would know which child it is. Default is None.
    return_pval (bool, optional): Boolean indicating whether to return p-values. Default is True.

    Returns:
    numpy.ndarray: Pseudo-bulk-by-position footprint score matrices for multiple modes.
    any: Extra information.
    """

    return_array = None
    if footprintRadius is None:
        footprintRadius = modes
    if flankRadius is None:
        flankRadius = modes

    modes = list(modes)

    for mode, r1, r2 in zip(modes, footprintRadius, flankRadius):
        result, mode = regionFootprintScore(
            region_ATAC,
            Tn5Bias,
            dispersionModels[str(mode)],
            r1,
            r2,
            mode,
            smoothRadius=smoothRadius,
            return_pval=return_pval,
        )
        if return_array is None:
            return_array = np.zeros((result.shape[0], len(modes), result.shape[-1]))
        return_array[:, modes.index(mode), :] = result

    if extra_info is not None:
        return return_array, extra_info
    return return_array


def _bigwig_footprint(
    insertion, bias, chrom, s, e, pad=0, extra=None, return_pval=True, smoothRadius=5
):
    """
    Calculates multiscale footprints from insertion and bias BigWig files.

    Parameters:
    insertion (pyBigWig.BigWig): The insertion BigWig file.
    bias (pyBigWig.BigWig): The bias BigWig file.
    chrom (str): The chromosome of the region.
    s (int): The start position of the region.
    e (int): The end position of the region.
    pad (int): The padding to apply to the region. Default is 0.
    extra (any): Extra data to be returned along with the footprints. Default is None.
    return_pval (bool): Whether to return p-values along with the footprints. Default is True.
    smoothRadius (int): The radius for smoothing the footprints. Default is 5.

    Returns:
    v (numpy.ndarray): The calculated multiscale footprints.
    extra (any): The extra data returned along with the footprints.
    """

    b = np.array(bias.values(chrom, s - pad, e + pad))
    b[np.isnan(b)] = 0.0

    a = np.array(insertion.values(chrom, s - pad, e + pad))
    a[np.isnan(a)] = 0.0
    v = multiscale_footprints(
        a[None],
        b,
        get_global_disp_models(),
        modes=np.arange(2, 101),
        smoothRadius=smoothRadius,
        return_pval=return_pval,
    )[0]
    if pad > 0:
        v = v[:, pad:-pad]
    if extra is not None:
        return v, extra
    return v