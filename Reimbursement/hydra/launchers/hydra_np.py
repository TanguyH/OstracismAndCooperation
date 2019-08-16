import sys
import numpy as np
import reimbursement

if __name__ == "__main__":
    # mu, sigma case analysis
    #mu = np.float64(sys.argv[1])
    #s = np.float64(sys.argv[2])

    # N case analysis
    s = np.float64(sys.argv[1])
    N = int(sys.argv[2])

    case = sys.argv[3]

    model = base_model.SocialLearningSim(N = N, M = 100, c = 1., r = 3., G = .7, B = .7, gamma = .7, beta = .7, sigma = 1., mu = 10**-3, s = 10, compulsory = True, SOP=False, case="no_punishment")
    model.runAnalyticalSim(N)
