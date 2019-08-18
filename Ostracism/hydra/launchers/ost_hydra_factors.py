import sys
import numpy as np
from ost_exportCSV import *

if __name__ == "__main__":
    N = 5
    M = 100
    c = 1.
    r = 3.
    G = .7
    B = .7
    gamma = .7
    beta = .7
    sigma = 1
    omega = -17
    mu = 10**-3
    s = 10

    case = sys.argv[1]
    compulsory = sys.argv[2]
    SOP = sys.argv[3]
    path = sys.argv[4]

    if compulsory == "True":
        compulsory = True
    else:
        compulsory = False
    if SOP == "True":
        SOP = True
    else:
        SOP = False

    factory = CSVFactory(N = N, M = M, c = c, r = r, G = G, B = B, gamma = gamma, beta = beta, sigma = sigma, omega = omega, mu = mu, s = s, compulsory = compulsory, SOP = SOP)
    #factory.generateDataFT(case)
    factory.generateDataImitation(case)

    #factory.generateDataGroupSize(case, "M")
    #factory.generateDataGroupSize(case, "N")

    #factory.generateDataPGGReturn(case, "r")
    #factory.generateDataPGGReturn(case, "c")

    #factory.generateDataEndowment(case)

    #factory.generateDataPunishmentFees(case, "pool")
    #factory.generateDataPunishmentFees(case, "peer")

    #factory.generateDataPunishmentFines(case, "pool")
    #factory.generateDataPunishmentFines(case, "peer")

    #factory.generateDataJointFees(case)
    #factory.generateDataJointFines(case)

    factory.generateDataExclusionPain(case)
