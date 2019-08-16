import sys
import numpy as np
import reimbursement

if __name__ == "__main__":
    # mu, sigma case analysis
    mu = np.float64(sys.argv[1])
    s = np.float64(sys.argv[2])

    # N case analysis
    #s = np.float64(sys.argv[1])
    #N = int(sys.argv[2])

    #case = sys.argv[3]

    model = reimbursement.SocialLearningSim(N = 5, M = 100, c = 1., r = 3., G = .7, B = .7, gamma = .7, beta = .7, sigma = 1., alpha = .7, mu = mu, s = s, compulsory = False, SOP=False, case="competition")
    model.runAnalyticalSim()
