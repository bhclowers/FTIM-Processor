import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from pyteomics import mzml
from scipy.interpolate import interp1d

#Developed by Elvin R. Cabrera and Brian H. Clowers @WSU

def getMS_v2(spectra):
    '''
    Return the full mass spectrum.
    '''
    mzIndex, intens = [], []
    #line below assumes all scans are the same in m/z dimension
    mzIndex = spectra[0]['m/z array']
    intens = np.zeros_like(spectra[0]['intensity array'])
    for i, s in enumerate(spectra):  # Assumes keys are numbers\
        if s['ms level'] == 1:
            intens += s['intensity array']

    mzIndex = np.array(mzIndex)
    intens = np.array(intens)

    return mzIndex, intens


def getXIC(spectra, mzVal, tol=0.05, secondsBool=True):
    '''
    Get the extracted mobility spectrum.
    '''
    tIndex, intens = [], []
    left, right = mzVal - tol, mzVal + tol
    for s in spectra:
        curTime = float(s['scanList']['scan'][0]['scan start time'])
        tIndex.append(curTime)

        curMZ = np.array(s['m/z array'], dtype=float)
        m = curMZ.searchsorted(left)
        n = curMZ.searchsorted(right)
        intens.append(s['intensity array'][m:n].sum())
    if secondsBool:
        return np.array(tIndex) * 60.0, np.array(intens)
    else:
        return np.array(tIndex), np.array(intens)


def loadOrbiMS_v2(fileName, scanStep=0.1):
    '''
    Loads an orbitrap mzML file for FT-IM Processing.
    As compared to a LTQ, the Orbitrap resolution is obviously much higher resolution which creates a processing problem for display.
    The data are stored in the mzML domain in sparse vector (i.e. non-evenly spaced points).  For display we need to fix that.
    To make this happen the high res data must be binned and depending on the resolution you choose the time for loading will scale.
    Smaller bins, longer load times.

    '''

    # get raw data dictionary
    rawData = scanOrbi(fileName, scanStep)

    x = rawData['mz_axis']
    y = np.zeros_like(rawData[1][1])
    for j, k in enumerate(rawData.keys()):
        if k != 'mz_axis':
            y += rawData[k][1]

    return x, y, rawData


def getXIC_Orbi(rawData, mzVal, tol=0.05, secondsBool=True):
    '''
    Get the extracted mobility spectrum.

    There are some subtle differences between the dictionaries between the orbi and LTQ datasets.
    '''
    tIndex, intens = [], []
    left, right = mzVal - tol, mzVal + tol
    for i, k in enumerate(rawData.keys()):
        if k != 'mz_axis':
            curTime = rawData[k][0]
            tIndex.append(curTime)

            curMZ = rawData['mz_axis']
            m = curMZ.searchsorted(left)
            n = curMZ.searchsorted(right)
            intens.append(rawData[k][1][m:n].sum())
    if secondsBool:
        #         return np.array(tIndex)*60.0, np.array(intens) #why divide by 60?
        return np.array(tIndex), np.array(intens)

    else:
        return np.array(tIndex), np.array(intens)


def scanOrbi(fname, step=0.1, saveBool=False):
    """

    Adapted from scripts written by McCabe and Laganowsky (TAMU)

    Read and extract data from mzML coverted from Thermo *.RAW. (e.g. pyteomics)

    Over the acquision, mass spectrum and time from each scan from the *.RAW will
    be extracted. Each scan is referred to as a scan number. This data will be formatted
    in a dictionary, which is necessary for FFT. Importantly, the data will be regrided
    based on input paramaters, and this grid should be constant among datasets in particular
    for averaging.

    Parameters
    ----------
    fname : str
        Path to *.mzML data
    step : float
        Step size for m/z grid
    saveBool : bool
        Save pickle file of data object for processed *.RAW

    Returns
    -------
    dict
        A dict keyed by scan_number with values containing time and m/z intensity array. Note
        that the m/z grid, used to re-grid all m/z intensity arrays for each scan, is stored in
        dict only once, which can be accessed using key "mz_axis".

    Do we need to add some check for the msLevel?

    """
    # print("Processing %s" % fname)
    msruns = [s for s in mzml.read(fname)]
    scans = np.arange(0, len(msruns))  # do we add 1 here
    mz_start = \
    msruns[0]['scanList']['scan'][0]['scanWindowList']['scanWindow'][0][
        'scan window lower limit'].real
    mz_end = \
    msruns[0]['scanList']['scan'][0]['scanWindowList']['scanWindow'][0][
        'scan window upper limit'].real

    # for storing data
    data = {}

    # set grid
    grid = np.arange(mz_start, mz_end, step)

    # for time, ms in zip(self.times, self.data):
    for i, s in enumerate(msruns):
        #         if i%100 == 0:
        #             print("Scan # %s"%i)
        # extract values
        ms = np.array(s['m/z array'])
        intensity = np.array(s['intensity array'])

        #         #time = self.msrun.scan_time_from_scan_name(s)
        time = s['scanList']['scan'][0]['scan start time'].real * 60

        #         # regrid mz axis to grid
        ms = regrid(grid, np.column_stack((ms, intensity)), False)

        #         # add to data
        data[i] = time, ms[::, 1]

    #     # store mz axis (or grid used)
    data["mz_axis"] = grid

    if saveBool:
        print("Not implemented")
        # now save the data
    #         f = open(fname+'.pkl', 'wb')
    #         pickle.dump(data,f)
    #         f.close()

    return data


def regrid(grid, data, debugBool=False):
    """
    Re-grid data to grid using interpolation

    Parameters
    ----------
    grid : np_array
        grid to interpolate values
    data : np_array
        An array containing m/z values [(m/z1, int1), (m/z2, int2), ...]

    Returns
    -------
    np_array
        A numpy array containing values regrided using interpolation.  The array is
        formatted the same is data parameter i.e. [(m/z1, int1), (m/z2, int2), ...]
    """
    if debugBool:
        print(len(data), len(grid))  # , data[:,0], data[:,1])

    f = interp1d(data[:, 0], data[:, 1], bounds_error=False, fill_value=0)
    #     f = interp1d(data[0], data[-1], bounds_error=False, fill_value=0)
    inty = f(grid)
    return np.column_stack((grid, inty))


def zero_padding(signal, padLen):
    '''
    This function adds zero padding to an input signal and returns the padded version of the signal.

    Parameters
    ------------
    signal : A signal over an evenly spaced time interval to add zero padding to
    padLen : decimal value of the front-end zero padding relative to the original signal
            --> e.g. '0.5' will give a pad that's 50% of the original signals length.

    Returns
    ------------
    newSig : zero-padded version of original signal

    Requires standard import of numpy.
    '''

    zerolen = int(len(signal) * padLen)
    pad = np.zeros((zerolen), dtype=np.float64)
    # newSig = np.concatenate((pad, signal))
    newSig = np.concatenate((signal, pad))

    # making sure it works as intended...
    #     print('Length of the zero-pad array {} \nLength of original signal {}'.format(len(pad), len(signal)))

    return newSig

def getAFT(freqAx, yFFT, deltaT, tOffset=0.1):
    '''
    Performs absorption mode FT

    Parameters
    ------------
    freqAx  :  X-axis returned from DFT_data
    yFFT    :  Magnitude Fourier transform vector returned from DFT_data
    deltaT  :  Corresponds to the starting frequency of the sweep divided by sweep rate of the experiment
    tOffset :  Manual offset for the time offset, Default value of 0.1 seconds


    Returns
    ------------
    aFT.real  : The real component of the absorption mode FT
    '''

    aFT = yFFT * np.exp(-1j * (deltaT + tOffset) * np.pi * freqAx)

    return aFT.real ** 2


def moCal(dtime, length=17.385, voltage=7860, T=297.7, P=690):
    '''
    Returns reduced mobility.

    Parameters
    ------------------
    dtime  : Drift time in (ms)
    length : Drift tube length (cm)
    voltage: Voltage across drift tube (V)
    T      : Temperature (K)
    P      : Pressure (torr)
    '''
    v = length / (dtime / 1000)  # cm/s
    E = voltage / length  # V/cm
    K = v / E
    K0 = K * (P / 760) * (273.15 / T)

    return K0









def XIC_preview(amp, windowBool = False, window='hanning',
                 padBool = False, padLen=0.5):
    '''
    Performs necessary manipulations to carry out a discrete Fourier Transform on experimental data.

    Parameters
    ------------
    amp   :  amplitude spectrum of interferogram
    window: (optional, defaults to 'hanning' window)
    padLen: (optional, defaults to 0.5)
             decimal value to determine zero padding length based on original signals length
             --> e.g. '0.5' will give a pad that's 50% of the original signals length.
    windowBool : (defaults to False) if windowing function is to be applied to signal
    padBool    : (defaults to False) if front-end zero padding is to be applied

    Returns
    ------------
    wY  : Amplitude array of new frequency domain signal
    '''

    # dictionary for different window functions
    windic = {'bartlett': np.bartlett, 'blackman': np.blackman,
              'hamming': np.hamming, 'hanning': np.hanning,
              'barthann': signal.barthann, 'bohman': signal.bohman,
              'nuttall': signal.nuttall,
              'parzen': signal.parzen, 'tukey': signal.tukey}

    if windowBool:
        wind = windic[window](len(amp))
        wY = amp * wind
    else:
        wY = amp

    if padBool:
        wY = zero_padding(wY, padLen)

    return wY


def multiFT_data(time, amp, sweepRate, method='FFT', window='hanning',
                 padLen=0.5, minBool=False,
                 windowBool=False, padBool=False, startFreq=5, tOffset=0.1):
    '''
    Performs necessary manipulations to carry out a discrete Fourier Transform on experimental data.

    Parameters
    ------------
    time  :  can either be in sec, or min - if min, minBool must be True
    amp   :  amplitude spectrum of interferogram
    method:  determines how data is handled.
                --> options: 'FFT', 'aFT'

    window: (optional, defaults to 'hanning' window)
             type of window to apply to the interpolated signal - available windows shown in 'windic' below

    padLen: (optional, defaults to 0.5)
             decimal value to determine front-end zero padding length based on original signals length
             --> e.g. '0.5' will give a pad that's 50% of the original signals length.

    minBool    : (defaults to False) True if signal is in minutes, otherwise it's assumed signal is in seconds
    windowBool : (defaults to False) if windowing function is to be applied to signal
    padBool    : (defaults to False) if front-end zero padding is to be applied

    startFreq  : Necessary input parameter for aFT method, value in Hz, needed for the input of the aFT function
    sweepRate  : Necessary input parameter for aFT method, value in Hz/s, needed for the input of the aFT function
    tOffset    : Necessary input parameter for aFT method, value in s, needed for the input of the aFT function

    Returns
    ------------
    X  : Frequency axis
    Y  : Amplitude array of new frequency domain signal

    Requires UnivariateSpline (scipy), scipy.signal, and numpy to be imported.
    '''

    #     print('{} points - {:.3f} seconds'.format(len(time), time[-1]))
    if minBool:
        time *= 60

    # dictionary for different window functions
    windic = {'bartlett': np.bartlett, 'blackman': np.blackman,
              'hamming': np.hamming, 'hanning': np.hanning,
              'barthann': signal.barthann, 'bohman': signal.bohman,
              'nuttall': signal.nuttall,
              'parzen': signal.parzen, 'tukey': signal.tukey}

    # picking length of new time vector based on original number of points

    # in the line below, I've chosen not to interpolate to a higher number of points
    # this keeps the sampling frequency the same and therefore the nyquist frequency doesnt change
    # interpolation is best left to a spline at the end or zero-padding the time domain signal


    n = int(len(time))  # keep same number of points.
    x_val = np.linspace(time[0], time[-1], n)

    # interpolating signal, smoothing value s=0 required for true interpolation
    ftspl = UnivariateSpline(time, amp, s=0)
    FTspl = ftspl(x_val)

    # ---recovering y-axis---
    # where a windowing function is or isn't applied
    if windowBool:
        wind = windic[window](len(FTspl))
        wY = FTspl * wind
    else:
        wY = FTspl

    if padBool:
        wY = zero_padding(wY, padLen)

    # ---creating x-axis vector---
    N = int(len(wY) / 2)
    mFac = 2 / n  # normalization factor

    W = np.fft.fftfreq(len(wY), np.diff(x_val).mean())
    X = W[:N]

    FY = np.fft.fft(wY)
    Y = np.abs(FY[:N]) * mFac

    # evaluating the splines atthis vector (2x the original # of points)
    X2 = np.linspace(X[0], X[-1], n * 2)

    # processing methods -------------------------------

    if method == 'FFT':
        return X, Y ** 2

    if method == 'aFT':
        deltaT = startFreq / sweepRate
        aFT = getAFT(X, FY[:N], deltaT,
                     tOffset)  # feed raw truncated FT vector and freq. axis
        aFT_spl = UnivariateSpline(X, aFT,
                                   s=0)  # cubic interpolation, add arg. "k=1" if want linear
        aFT_spec = aFT_spl(
            X2)  # eval spl. at NEW x vector with more points for better fits
        return X2, aFT_spec  # already squared








