import numpy as np
from scipy import constants

#Developed by Brian H. Clowers et al. @WSU

def calcThermalVelocity(temp, mu):
    '''
    temp is in K
    mu = reduced mass
    return value is meter/sec
    '''
    kb = 1.38065E-16  #erg/K
    amu = 1.660539E-24  #gm
    vt = (8 * kb * temp / (np.pi * mu * amu))
    vt = np.sqrt(vt) / 100
    return vt


def calcAlpha(m, M, fc):
    mhat = m / (m + M)
    Mhat = M / (m + M)
    alpha = (2 / 3.0) * (1 + mhat * fc + Mhat * (1 - fc))
    return alpha


def calcTransverseVelCoeff(m, M):
    '''
    Also known as $beta_MT$
    '''
    mhat = m * 1.0 / (m + M)
    return np.sqrt(2.0 / mhat / (1 + mhat))


def calcNumberDensity(press, temp,
                      No=2.68677E25):  # change to 2.68677E19 if you want cm^-3
    '''
    Calculate molecular number density
    Pressure is in torr
    Temp is in Kelvin
    return value is in units of m^-3
    '''
    return No * (273.15 / temp) * (press / 760.0)


def ccsFromDriftTime(voltage,
                     length,
                     pressure,
                     temp,
                     driftTime,
                     charge,
                     ionMass,
                     gasMass,
                     debug=False):
    '''
    ccs is in square angstroms
    pressure is in torr
    length is in cm
    Temp is in Kelvin
    ionMass and gasMass are in amu
    ccs in in squared length units
    '''
    kb = constants.k
    LC = 2.6867774e25  #Loschmidt Constant in m^-3
    massConv = 1.660539E-24  #gm/amu
    length /= 100.0  #in m
    numberDensity = calcNumberDensity(pressure, temp)

    #<vT>=(8kT/πμ)1/2 (m s-1)
    reducedMass = ((ionMass * gasMass) / (ionMass + gasMass))
    thermalVelocity = calcThermalVelocity(temp, reducedMass)  #m/s
    vd = length / driftTime
    if debug:
        print("Drift Velocity in m/s:", vd)
    eField = voltage / (length)  #V/m
    numerator = 0.75 * charge * constants.e * eField / numberDensity / 1.0E-17  #Townsend factor
    denominator = reducedMass * massConv * thermalVelocity * vd
    ccs = numerator / denominator * 1E6  #scaling factor for units
    if debug:
        print("E-Field: ", eField / 100, 'V/cm')
        print("Drift Velocity", vd, "m/s")
        print("Thermal Velocity: ", thermalVelocity)
        print("Number Density: ", numberDensity)
        print("Numerator: ", numerator)
        print("Denominator: ", denominator)

    return ccs  # in square angstroms


def calcCorrectedCCS(ionMass, ionCharge, gasMass, driftLength, driftPotential,
                     driftTime, gasPress, gasTemp, fcList,
                     debug=False):
    '''
    ccs is in square angstroms
    pressure is in torr
    length is in cm
    Temp is in Kelvin
    ionMass and gasMass are in amu
    drift time is in seconds
    potential is in Volts
    '''

    # Scale some of the input values to get them in SI units
    driftLength /= 100.0  # in m
    massConv = 1.660539E-24  # gm/amu

    reducedMass = ((ionMass * gasMass) / (ionMass + gasMass))
    vt = calcThermalVelocity(gasTemp, reducedMass)  # m/s
    vd = driftLength / driftTime
    velRatio = vd / vt
    fc = np.interp(velRatio, fcList[0], fcList[1])

    alpha = calcAlpha(ionMass, gasMass, fc)
    beta = calcTransverseVelCoeff(ionMass, gasMass)  # calcBeta(velRatio)
    numDens = calcNumberDensity(gasPress, gasTemp, No=2.68677E25)

    ccsRaw = ccsFromDriftTime(driftPotential, driftLength, gasPress, gasTemp,
                              driftTime, ionCharge, ionMass, gasMass, False)

    eField = driftPotential / (driftLength)  # V/m
    corrCCS = ccsRaw * 1.0 / np.sqrt(
        1 + ((beta / alpha) ** 2) * ((vd / vt) ** 2))

    if debug:
        print("E-Field: ", eField / 100, 'V/cm')
        print("Drift Velocity: ", vd)
        print("Thermal Velocity: ", vt)
        print("Number Density: ", numDens)
        print('fc: ', fc)
        print("Alpha: ", alpha)
        print("Beta: ", beta)
        #         print("Zeta: ", zeta)
        print("E/N :",
              eField / numDens * 1E17 * 1E4)  # Convert to townsend and get rid of cm

    return corrCCS * 1E-4  # scaling factor for dimensions
