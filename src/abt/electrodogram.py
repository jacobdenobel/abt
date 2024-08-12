# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:29:42 2019

@author: beimx004
"""

import numpy as np

from scipy.signal import lfilter
from scipy.interpolate import interp1d

from .defaults import DEFAULT_CHANNEL_ORDER


def f120(
    ampIn,
    nChan=15,
    pulseWidth=18,
    outputFs=[],
    channelOrder=None,
    cathodicFirst=True,
    weights=None,
    **kwargs,
):
    """
    elGram = f120ElectrodogramFunc(par, ampIn)
    Generate scope-like electrodogram from matrix of F120 amplitude frame.
    Amplitude frames are expected to represent the amplitude( pair)s for each
    channel by a pair of consecutive rows each (as provided e.g. by
    F120MappingUnit)

    Input:
      par - parameter object/struct
      ampIn - 2*nChan x nFtFrames matrix of stimulation amplitudes [uA]

    Fields of par:
      channelOrder - 1 x nChan vector defining the firing order among channels
                     [1..nChan, unique] [[1 5 9 13 2 6 10 14 3 7 11 15 4 8 12]]
      outputFs - output sampling frequency; [] for native FT rate  [Hz] [[]]
                 (resampling is done using zero-order hold method)
      cathodicFirst - start biphasic pulse with cathodic phase [bool] [true]
      enablePlot - generate electrodogram plot? [bool]
      colorScheme - color scheme for plot; [1..4] 1/2 more subdued, 3/4 more strident colors; odd/even affects color order

    Output:
      elGram - 16 x nSamp matrix of electrode current flow; [uA]

    Copyright (c) 2019-2020 Advanced Bionics. All rights reserved.
    """
    nFrameFt = ampIn.shape[1]
    phasesPerCyc = 2 * nChan
    dtIn = phasesPerCyc * pulseWidth * 1e-6
    durIn = nFrameFt * dtIn

    if channelOrder is None:
        channelOrder = DEFAULT_CHANNEL_ORDER

    assert nChan == 15, "only 15-channel strategies are supported."
    assert channelOrder.shape[0] == nChan, "length(channelOrder) must match nChan"

    nFrameOut = nFrameFt * phasesPerCyc

    idxLowEl = np.arange(nChan)
    idxHighEl = np.arange(nChan) + 1
    nEl = 16

    elGram = np.zeros((nEl, nFrameOut))
    weights_matrix = np.zeros((nEl, nFrameOut))
        
    for iCh in np.arange(nChan):
        phaseOffset = 2 * (channelOrder[iCh] - 1)
        elGram[idxLowEl[iCh], phaseOffset::phasesPerCyc] = ampIn[2 * iCh, :]
        elGram[idxHighEl[iCh], phaseOffset::phasesPerCyc] = ampIn[2 * iCh + 1, :]
        
        weights_matrix[idxLowEl[iCh], phaseOffset::phasesPerCyc] = (
            weights[iCh, :]
        )
        weights_matrix[idxHighEl[iCh], phaseOffset::phasesPerCyc] = (
            weights[iCh + nChan, :]
        )

    if cathodicFirst:
        kernel = np.array([-1, 1])
    else:
        kernel = np.array([1, -1])

    elGram = lfilter(kernel, 1, elGram)

    if outputFs:
        dtOut = 1 / outputFs
        tPhase = np.arange(nFrameOut) * pulseWidth * 1e-6
        tOut = np.arange(np.floor(durIn / dtOut)) * dtOut
        fElGram = interp1d(tPhase, elGram, kind="previous", fill_value="extrapolate")
        elGram = fElGram(tOut)
    else:
        tOut = np.arange(nFrameOut) * pulseWidth * 1e-6

    return elGram, weights_matrix
