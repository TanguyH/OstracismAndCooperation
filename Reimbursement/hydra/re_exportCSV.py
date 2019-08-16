import numpy as np
from reimbursement import *
import time

class CSVFactory:
    def __init__(self, N, M, c, r, G, B, gamma, beta, sigma, alpha, mu, s, compulsory, SOP):
        self.__N = N
        self.__M = M
        self.__c = c
        self.__r = r
        self.__G = G
        self.__B = B
        self.__gamma = gamma
        self.__beta = beta
        self.__sigma = sigma
        self.__mu = mu
        self.__s = s
        self.__alpha = alpha
        self.__compulsory = compulsory
        self.__SOP = SOP

    #---------------------------------------------------------------------------
    #   GETTERS
    #---------------------------------------------------------------------------
    def isCompulsory(self):
        return self.__compulsory

    def SOPallowed(self):
        return self.__SOP

    #---------------------------------------------------------------------------
    #   FILE GENERATORS
    #---------------------------------------------------------------------------
    def generateFileName(self, case):
        # file name formatting
        name = case
        if self.__SOP and case != "no_punishment":
            name = "SOP_{}".format(name)

        if self.__compulsory:
            name += "_compulsory"
        else:
            name += "_voluntary"
        return name

    def createDataFile(self, case, factor):
        context = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
        available_strategies = context.getAvailableStrategies()

        # file name formatting
        name = self.generateFileName(case)
        data_file = open("reimbursement/re_data/{}/{}.csv".format(factor, name), "w")

        # format header
        header = "{}".format(factor)
        for strategy in available_strategies:
            header += ",{}".format(strategy)
        data_file.write("{}\n".format(header))

        return data_file

    #---------------------------------------------------------------------------
    #   WRITER
    #---------------------------------------------------------------------------
    def calcAndWrite(self, sim, factor, file):
        trans = sim.computeTransitionMatrix()
        stationary = sim.computeStationaryDistribution(trans)

        res = "{}".format(factor)
        for distribution in stationary:
            res += ",{}".format(distribution)
        res += "\n"

        file.write(res)

    #---------------------------------------------------------------------------
    #   DATA GENERATORS
    #---------------------------------------------------------------------------
    def generateDataImitation(self, case, factor="s"):

        data_file = self.createDataFile(case, factor)
        imitation_rates = np.logspace(-4., 4., 50)

        for s in imitation_rates:
            sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = s, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
            self.calcAndWrite(sim, s, data_file)
        data_file.close()

    def generateDataFT(self, case):

        context = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
        available_strategies = context.getAvailableStrategies()

        # file name formatting
        name = self.generateFileName(case)
        data_file_unconditional = open("data/times/{}_{}.csv".format("unconditional", name), "w")
        data_file_conditional = open("data/times/{}_{}.csv".format("conditional", name), "w")
        imitation_rates = np.logspace(-4., 4., 50)

        # format header
        header = "s"
        for resident in available_strategies:
            for invader in available_strategies:
                if resident != invader:
                    header += ",{}{}".format(resident, invader)
        data_file_unconditional.write("{}\n".format(header))
        data_file_conditional.write("{}\n".format(header))

        for s in imitation_rates:
            sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = s, case = case, compulsory = self.__compulsory, SOP=self.__SOP)

            uncond_res = "{}".format(s)
            cond_res = "{}".format(s)
            for resident in available_strategies:
                for invader in available_strategies:
                    if resident != invader:
                        uncond_res += ",{}".format(sim.unconditionalFixationTime(resident, invader))
                        cond_res += ",{}".format(sim.conditionalFixationTime(resident, invader))
            uncond_res += "\n"
            cond_res += "\n"

            data_file_unconditional.write(uncond_res)
            data_file_conditional.write(cond_res)
        data_file_conditional.close()
        data_file_unconditional.close()

    def generateDataGroupSize(self, case, factor = "N"):
        data_file = self.createDataFile(case, factor)

        if factor == "N":
            sample_sizes = [i for i in range(2, self.__M+1)]
            for N in sample_sizes:
                sim = SocialLearningSim(N = N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
                self.calcAndWrite(sim, N, data_file)
        else:
            population_sizes = [i for i in range(5, 501, 5)]
            for M in population_sizes:
                sim = SocialLearningSim(N = self.__N, M = M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
                self.calcAndWrite(sim, M, data_file)
        data_file.close()

    def generateDataPGGReturn(self, case, factor="c"):
        data_file = self.createDataFile(case, factor)

        if factor == "c":
            contributions = [np.round(i,1) for i in np.arange(0.9, 10.1, .1)]
            for c in contributions:
                sim = SocialLearningSim(N = self.__N, M = self.__M, c = c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
                self.calcAndWrite(sim, c, data_file)
        else:
            multiplication_factors = [np.round(i,1) for i in np.arange(2.8, 10.1, .1)]
            for r in multiplication_factors:
                sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
                self.calcAndWrite(sim, r, data_file)
        data_file.close()

    def generateDataEndowment(self, case, factor="sigma"):
        data_file = self.createDataFile(case, factor)
        endowments = [np.round(i,2) for i in np.arange(0, 1.3, .05)]

        for sigma in endowments:
            sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
            self.calcAndWrite(sim, sigma, data_file)
        data_file.close()

    def generateDataPunishmentFees(self, case, target="pool"):
        if target == "pool":
            factor = "G"
        else:
            factor = "gamma"

        data_file = self.createDataFile(case, factor)
        costs = [np.round(i,2) for i in np.arange(0, 1, .05)]

        for cost in costs:
            if factor == "G":
                sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = cost, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
            else:
                sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = cost, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
            self.calcAndWrite(sim, cost, data_file)
        data_file.close()

    def generateDataPunishmentFines(self, case, target="pool"):
        if target == "pool":
            factor = "B"
        else:
            factor = "beta"

        data_file = self.createDataFile(case, factor)
        fines = [np.round(i,2) for i in np.arange(0, 2.55, .05)]

        for fine in fines:
            if factor == "B":
                sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = fine, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
            else:
                sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = fine, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
            self.calcAndWrite(sim, fine, data_file)
        data_file.close()

    def generateDataJointFees(self, case, factor = "fees"):
        data_file = self.createDataFile(case, factor)
        fees = [np.round(i,2) for i in np.arange(0, 1, .05)]

        for fee in fees:
            sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = fee, B = self.__B, gamma = fee, beta = self.__beta, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
            self.calcAndWrite(sim, fee, data_file)
        data_file.close()

    def generateDataJointFines(self, case, factor = "fines"):
        data_file = self.createDataFile(case, factor)
        fines = [np.round(i,1) for i in np.arange(0, 5., .1)]

        for fine in fines:
            sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = fine, gamma = self.__gamma, beta = fine, sigma = self.__sigma, alpha = self.__alpha, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
            self.calcAndWrite(sim, fine, data_file)
        data_file.close()

    def generateDataRestorationRate(self, case, factor = "alpha"):
        data_file = self.createDataFile(case, factor)
        restoration_rates = [np.round(i,2) for i in np.arange(0., 1.26, .01)]

        for rate in restoration_rates:
            sim = SocialLearningSim(N = self.__N, M = self.__M, c = self.__c, r = self.__r, G = self.__G, B = self.__B, gamma = self.__gamma, beta = self.__beta, sigma = self.__sigma, alpha = rate, mu = self.__mu, s = np.Infinity, case = case, compulsory = self.__compulsory, SOP = self.__SOP)
            self.calcAndWrite(sim, rate, data_file)
        data_file.close()

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
    alpha = .7
    mu = 10**-3
    s = 10
    compulsory = True
    SOP = False

    for case in ["no_punishment", "pool", "peer", "competition"]:
        for c_val in [True, False]:
            for sop_val in [True, False]:
                factory = CSVFactory(N = N, M = M, c = c, r = r, G = G, B = B, gamma = gamma, beta = beta, sigma = sigma, alpha = alpha, mu = mu, s = s, compulsory = c_val, SOP = sop_val)
                #factory.generateDataFT(case)
                factory.generateDataGroupSize(case, "N")
                factory.generateDataPGGReturn(case, "r")
                factory.generateDataEndowment(case)
                factory.generateDataPunishmentFees(case, "pool")
                factory.generateDataPunishmentFees(case, "peer")
                factory.generateDataPunishmentFines(case, "pool")
                factory.generateDataPunishmentFines(case, "peer")
                factory.generateDataJointFees(case)
                factory.generateDataJointFines(case)
                factory.generateDataRestorationRate(case)


    #factory = CSVFactory(N = N, M = M, c = c, r = r, G = G, B = B, gamma = gamma, beta = beta, sigma = sigma, mu = mu, s = s, compulsory = compulsory, SOP = SOP)
    #factory.generateDataImitation("no_punishment")
    #factory.generateDataFT("peer")
    #factory.generateDataGroupSize("no_punishment")
