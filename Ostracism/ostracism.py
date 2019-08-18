from scipy import linalg as LA
import numpy as np
import random
import copy as cp
import time
import operator as op
from functools import *
from agent import Agent
import os

COOPERATE = "X"
DEFECT = "Y"

POOLOSTRACIZE = "R"
PEEROSTRACIZE = "S"

STRATEGIES = [COOPERATE, DEFECT, POOLOSTRACIZE, PEEROSTRACIZE]

MAX_EXP = 709.78271289338397
INFINITE_FLAG = 1350
MAX_LD = 1.18973149536e+4932

NO_PUNISHMENT = "no_punishment"
POOL_OSTRACISM = "pool"
PEER_OSTRACISM = "peer"
COMPETITION = "competition"

def comb(n, k):
    k = min(k, n-k)
    numer = reduce(op.mul, range(n, n-k, -1), 1)
    denom = reduce(op.mul, range(1, k+1), 1)
    return numer//denom

class SocialLearningSim:
    def __init__(self, N, M, c, r, G, B, gamma, beta, sigma, mu, s, omega, compulsory = False, SOP = False, case="no_punishment"):

        # population information
        self.__sample_size = N
        self.__population_size = M

        # payoff fundamentals
        self.__contribution = c
        self.__pot_factor = r
        self.__endowment = sigma

        # punishment parameters
        self.__pool_punish_fee = G
        self.__pool_punish_fine = B
        self.__peer_punish_fee = gamma
        self.__peer_punish_fine = beta

        # learning characteristics
        self.__exploration_rate = mu
        self.__imitation_strength = s

        # ostracism specific factors
        self.__exclusion_pain = omega

        # participation status
        self.__compulsory = compulsory
        self.__SOP = SOP

        # case status
        self.__case = case

        # working vector
        self.__frequencies = [0 for strategy in STRATEGIES]

    #---------------------------------------------------------------------------
    # GETTERS
    #---------------------------------------------------------------------------
    def getSampleSize(self):
        return self.__sample_size

    def getPopulationSize(self):
        return self.__population_size

    def getContribution(self):
        return self.__contribution

    def getPotFactor(self):
        return self.__pot_factor

    def getEndowment(self):
        return self.__endowment

    def getPoolFee(self):
        return self.__pool_punish_fee

    def getPeerFee(self):
        return self.__peer_punish_fee

    def getPoolFine(self):
        return self.__pool_punish_fine

    def getPeerFine(self):
        return self.__peer_punish_fine

    def getExplorationRate(self):
        return self.__exploration_rate

    def getImitationStrength(self):
        return self.__imitation_strength

    def getExclusionPain(self):
        return self.__exclusion_pain

    def isCompulsory(self):
        return self.__compulsory

    def SOPallowed(self):
        return self.__SOP

    def getCase(self):
        return self.__case

    def getAvailableStrategies(self):
        available_strategies = [COOPERATE, DEFECT]

        # manage V, W strategy
        case = self.getCase()
        if case == POOL_OSTRACISM:
            available_strategies.append(POOLOSTRACIZE)
        elif case == PEER_OSTRACISM:
            available_strategies.append(PEEROSTRACIZE)
        elif case == COMPETITION:
            available_strategies.append(POOLOSTRACIZE)
            available_strategies.append(PEEROSTRACIZE)

        return available_strategies

    #---------------------------------------------------------------------------
    # WORKING VECTOR METHODS
    #---------------------------------------------------------------------------
    def getFrequency(self, strategy):
        return self.__frequencies[STRATEGIES.index(strategy)]

    def setFrequency(self, strategy, freq):
        self.__frequencies[STRATEGIES.index(strategy)] = freq

    def clearFrequencies(self):
        self.__frequencies = [0 for strategy in STRATEGIES]

    #---------------------------------------------------------------------------
    # SIM METHODS
    #---------------------------------------------------------------------------
    def getFrequencies(self):
        X = self.getFrequency(COOPERATE)
        Y = self.getFrequency(DEFECT)
        R = self.getFrequency(POOLOSTRACIZE)
        S = self.getFrequency(PEEROSTRACIZE)
        return X, Y, R, S

    def probaNoY(self):
        M = self.getPopulationSize()
        N = self.getSampleSize()
        Y = self.getFrequency(DEFECT)
        return comb(M - Y - 2, N - 2) / comb(M - 2, N - 2)

    def probaNoR(self):
        M = self.getPopulationSize()
        N = self.getSampleSize()
        R = self.getFrequency(POOLOSTRACIZE)
        return comb(M - R - 2, N - 2) / comb(M - 2, N - 2)

    def probaOstracizedY(self):

        # probability of being ostracized = all samples containing Y and R or S / all samples
        # 1 - samples not containing Y - samples not containing R and S

        tot_proba = 0
        M = self.getPopulationSize()
        N = self.getSampleSize()

        X, Y, R, S = self.getFrequencies()
        total = comb(M, N)

        if Y and (R or S):
            # possible other persons in sample
            for y in range(1, N):
                #print("{} defectors - {} others".format(y, N-y))
                for rs in range(0, N-y+1):
                    #print("\tof whom {} punishers and {} cooperators".format(rs, N-y-rs))
                    #tot_proba += (comb(X, i) * comb(R + S, j) * comb(X, N-i-j) / comb(M, N))


                    # if both punishers
                    if R and S:
                        for r in range(0, rs+1):
                            proba = (comb(X, N-y-rs) * comb(R, r) * comb(S, rs-r) * comb(Y, y))/total
                            tot_proba += proba
                    # pool = ostracized
                    elif R:
                        proba = (comb(X, N-y-rs) * comb(R, rs) * comb(Y, y))/total
                        tot_proba += proba

                    # peer + defector = ostaracized
                    elif S:
                        proba = (comb(X, N-y-rs) * comb(S, rs) * comb(Y, y))/total
                        tot_proba += proba

        return tot_proba

    def probaOstracizedX(self):
        tot_proba = 0
        X, Y, R, S = self.getFrequencies()
        M, N = self.getPopulationSize(), self.getSampleSize()

        if X and (R or S):

            #
            #   CHANGES TO APPLY :  X are ostracized by pool in ANY CASE
            #                       X are ostracized by peer only if Y present
            #

            # number of samples
            total =  comb(M, N)

            # possible other persons in sample
            for x in range(1, N):
                #print("{} cooperators - {} others".format(x, N-x))
                for rs in range(0, N-x+1):
                    #print("\tof whom {} punishers and {} defectors".format(rs, N-x-rs))
                    #tot_proba += (comb(X, i) * comb(R + S, j) * comb(X, N-i-j) / comb(M, N))

                    # if both punishers
                    if R and S:
                        for r in range(0, rs+1):
                            #print("\t\tof whom {} pool and {} peer".format(r, rs-r))

                            # if pool present = ostracized
                            if r:
                                proba = (comb(X, x) * comb(R, r) * comb(S, rs-r) * comb(Y, N-x-rs))/total
                                tot_proba += proba
                            # otherwise need peer + defector
                            elif rs-r and N-x-rs:
                                proba = (comb(X, x) * comb(R, r) * comb(S, rs-r) * comb(Y, N-x-rs))/total
                                tot_proba += proba
                    # pool = ostracized
                    elif R:
                        proba = (comb(X, x) * comb(R, rs) * comb(Y, N-x-rs))/total
                        tot_proba += proba

                    # peer + defector = ostaracized
                    elif S and rs and N-x-rs:
                        proba = (comb(X, x) * comb(S, rs) * comb(Y, N-x-rs))/total
                        tot_proba += proba

        return tot_proba


    def getRawPayoffX(self):
        c = self.getContribution()
        r = self.getPotFactor()
        sig = self.getEndowment()
        M, N = self.getPopulationSize(), self.getSampleSize()
        X, Y, R, S = self.getFrequencies()

        payoff = c * (r * ((M - Y - 1)/(M - 1)) - 1)
        return payoff

    def getPayoffX(self):
        payoff = self.getRawPayoffX()

        if self.SOPallowed():
            if self.getFrequency(POOLOSTRACIZE) or (self.getCase() == COMPETITION and self.getFrequency(POOLOSTRACIZE)):
                proba_ostracized = self.probaOstracizedX()
                if proba_ostracized > 1:
                    proba_ostracized = 1
            else:
                proba_ostracized = 0

            payoff = (proba_ostracized * self.getExclusionPain()) + ((1 - proba_ostracized) * payoff)
        return payoff

    def getRawPayoffY(self):
        c, r = self.getPotFactor(), self.getContribution()
        sig = self.getEndowment()
        M, N = self.getPopulationSize(), self.getSampleSize()
        Y = self.getFrequency(DEFECT)

        payoff = c * r *((M - Y)/(M - 1))

        return payoff

    def getPayoffY(self):
        payoff = self.getRawPayoffY()
        proba_excluded = self.probaOstracizedY()

        if proba_excluded > 1:
            proba_excluded = 1
        payoff = (proba_excluded * self.getExclusionPain()) + ((1-proba_excluded) * payoff)

        return payoff

    def getPayoffR(self):
        G = self.getPoolFee()
        payoff = self.getRawPayoffX() - G
        return payoff

    def getPayoffS(self):
        payoff = self.getRawPayoffX()
        M, N = self.getPopulationSize(), self.getSampleSize()
        sample_proba = (N-1) / (M-1)
        avg_fine = self.getPeerFee() * sample_proba * self.getFrequency(DEFECT)

        payoff = payoff - avg_fine

        if self.SOPallowed():
            additional_fine = self.getPeerFee() * sample_proba * self.getFrequency(COOPERATE) * (1 - self.probaNoY())
            payoff = payoff - additional_fine

            if self.getCase() == COMPETITION:
                proba_no_pool = self.probaNoR()
                payoff = (proba_no_pool * payoff) + (self.getExclusionPain() * (1 - proba_no_pool))

        return payoff

    def getPayoff(self, strategy):
        if strategy == COOPERATE:
            return self.getPayoffX()
        elif strategy == DEFECT:
            return self.getPayoffY()
        elif strategy == POOLOSTRACIZE:
            return self.getPayoffR()
        else:
            return self.getPayoffS()

    #---------------------------------------------------------------------------
    #   REQUIRED FOR STATIONARY DISTRIBUTIONS
    #---------------------------------------------------------------------------
    def computeFermiExp(self, payoff_1, payoff_2):
        raw_exp = self.getImitationStrength() * (payoff_1 - payoff_2)
        if np.isnan(raw_exp):
            raw_exp = 0
        elif raw_exp > MAX_EXP:
            raw_exp = MAX_EXP
        return raw_exp

    def Fermi(self, payoff_1, payoff_2):
        raw_exp = self.computeFermiExp(payoff_1, payoff_2)
        raw_exp *= -1
        return 1. / (1. + np.exp(raw_exp))

    def FermiNeg(self, payoff_1, payoff_2):
        raw_exp = self.computeFermiExp(payoff_1, payoff_2)
        return 1. / (1. + np.exp(raw_exp))

    def fixResidentInvaderFreq(self, resident, invader, i):
        M = self.getPopulationSize()
        self.clearFrequencies()
        self.setFrequency(resident, M-i)
        self.setFrequency(invader, i)

    def rho(self, resident, invader):
        sum = np.longdouble(0.0)
        for q in range(1, self.getPopulationSize()):
            sub_sum = np.longdouble(0.0)
            for invader_qty in range(1, q + 1):
                self.fixResidentInvaderFreq(resident, invader, invader_qty)

                res_payoff = self.getPayoff(resident)
                inv_payoff = self.getPayoff(invader)

                payoff_diff = res_payoff - inv_payoff
                sub_sum += payoff_diff

            if(sub_sum != 0):
                sub_sum *= self.getImitationStrength()

            if(sub_sum > MAX_EXP):
                sub_sum = MAX_EXP

            sum += np.exp(sub_sum)

        return np.round(1. / (1. + sum), 10)

    def computeTransitionMatrix(self, mu = False):
        """
        Based on the existing strategies, the function computes the transition
        matrix for the homogenous states
        """

        available_strategies = self.getAvailableStrategies()
        t_matrix = np.matrix([[0. for strategy in available_strategies] for strategy in available_strategies])

        for i, resident in enumerate(available_strategies):
            for j, invader in enumerate(available_strategies):
                if (i != j):
                    fixation_proba = self.rho(resident, invader)
                    if(mu):
                        fixation_proba *= self.getExplorationRate()
                    t_matrix[i, j] = 1./(len(available_strategies) - 1) * fixation_proba

        # calculate diagonal elements
        for i in range(len(available_strategies)):
            t_matrix[i, i] = 1 - t_matrix[i].sum(axis = 1)

        return t_matrix

    def norm(self, vector):
        a = vector.real
        b = vector.imag
        norm = np.sqrt(a**2 + b**2)
        return norm

    def computeStationaryDistribution(self, trans_mat):
        """
        Based on the transition matrix, the function obtains a Markovian Matrix
        with wich it can compute the eigenvectors and eigenvalues. Next, it finds
        the eigenvector to the eigenvalue 1 and subsequently normalizes it to
        provide the stationary distribution
        """
        t_matrix = np.transpose(trans_mat)
        w, v = np.linalg.eig(t_matrix)

        # extract eigenvalues
        w = w.tolist()
        norm_w = []
        for eigenvalue in w:
            norm = self.norm(eigenvalue)
            norm_w.append(round(norm,5))

        # find eigenvector to eigenvalue 1
        vector_index = norm_w.index(1.)

        # normalize eigenvector
        eigenvector = v[:,vector_index].tolist()
        norm_eigenvector = []
        for component in eigenvector:
            component = component[0]
            norm = self.norm(component)
            norm_eigenvector.append(norm)

        norm_eigenvector = norm_eigenvector / sum(norm_eigenvector)

        return norm_eigenvector

    #---------------------------------------------------------------------------
    #   REQUIRED FOR SIMULATIONS
    #---------------------------------------------------------------------------
    def saveResults(self, res_vector, name, factor="mu"):
        case = "voluntary"
        if self.isCompulsory():
            case = "compulsory"

        punish_flag = ""
        if(self.SOPallowed()):
            punish_flag = "SOP_"

        file_name = "ostracism/{}{}_{}.csv".format(punish_flag, name, case)

        # does not exist : create file
        file = open("{}".format(file_name), "a+")

        # write header only if file empty
        if os.stat(file_name).st_size == 0:
            header = "{}".format(factor)
            for strategy in self.getAvailableStrategies() + [OSTRACIZED]:
                header += ",{}".format(strategy)
            header += "\n"
            file.write(header)

        # write header line
        if factor == "mu":
            res_line = "{}".format(self.getExplorationRate())
        if factor == "s":
            res_line = "{}".format(self.getImitationStrength())
        elif factor == "N":
            res_line == "{}".format(self.getSampleSize())
        elif factor == "c":
            res_line == "{}".format(self.getContribution())
        elif factor == "r":
            res_line == "{}".format(self.getPotFactor())
        elif factor == "sigma":
            res_line == "{}".format(self.getEndowment())

        # write results
        for res in res_vector:
            res_line += ",{}".format(res)
        res_line += "\n"
        file.write(res_line)

        # close file
        file.close()

    def runAnalyticalSim(self, sim_rounds = 10**7, factor="mu"):
        """
        sim where 10**7 rounds are played by agents (circular)
        TO COME: not circular (each player 10**7 times)
        """
        # create simulation agents
        M = self.getPopulationSize()
        N = self.getSampleSize()

        available_strategies = self.getAvailableStrategies()
        sim_agents = [Agent(available_strategies) for i in range(M)]
        tot_count = [0 for strategy in available_strategies]

        # count strategies in current population
        strat_count = [0 for strategy in available_strategies]
        for sim_agent in sim_agents:
            strat_count[available_strategies.index(sim_agent.getStrategy())] += 1

        # repeat 10 million times
        for i in range(sim_rounds):

            # handle each agent
            for focal_player in sim_agents:

                # update frequencies for avg payoffs
                self.clearFrequencies()
                for i, strategy in enumerate(available_strategies):
                    self.setFrequency(strategy, strat_count[i])

                # option 1: random switch strategy
                mu_proba = np.random.random()
                if mu_proba <= self.getExplorationRate():
                    strat_count[available_strategies.index(focal_player.getStrategy())] -= 1
                    focal_player.switchToOtherAvailableStrategy()
                    strat_count[available_strategies.index(focal_player.getStrategy())] += 1

                # option 2: choose model to (maybe) imitate
                else:
                    # select model player
                    model_player_index = np.random.randint(0, M-1)
                    while model_player_index == sim_agents.index(focal_player):
                        model_player_index = np.random.randint(0, M-1)
                    model_player = sim_agents[model_player_index]

                    # define imitation outcome
                    proba_copy = self.Fermi(self.getPayoff(model_player.getStrategy()), self.getPayoff(focal_player.getStrategy()))
                    proba_event = np.random.random()
                    if proba_event <= proba_copy:
                        strat_count[available_strategies.index(focal_player.getStrategy())] -= 1
                        focal_player.setStrategy(model_player.getStrategy())
                        strat_count[available_strategies.index(focal_player.getStrategy())] += 1

            # remember population strategies
            for i in range(len(tot_count)):
                tot_count[i] += strat_count[i]

        # obtain final frequency
        for i in range(len(strat_count)):
            strat_count[i] = strat_count[i] / M

        # obtain total frequency
        for i, strategy in enumerate(available_strategies):
            tot_count[i] = tot_count[i] / (sim_rounds * M)

        # export to file: strat_count (enables comparison of both results)
        self.saveResults(tot_count, "{}".format(self.getCase()), factor)

if __name__ == "__main__":
    SLS = SocialLearningSim(N = 5,                      # only extreme case of N = M
                            M = 100,                    # size of M seems to impact levels
                            c = 1.,                     # seems to have impact
                            r = 3.,                     # no impact at all
                            G = .7,
                            B = .7,
                            gamma = .7,
                            beta = .7,
                            sigma = 1.,                 # only extreme case of no payoff
                            mu = 10**-3,                # analysis done: impact clear
                            omega = -36.0,                # ostracized payoff (pain of ostracism)
                            s = np.Infinity,            # analysis done: impact clear
                            compulsory = False,
                            SOP = True,
                            case = COMPETITION)
    print(SLS.getAvailableStrategies())
    matrix = SLS.computeTransitionMatrix()
    print(matrix)
    s_t = SLS.computeStationaryDistribution(matrix)
    print(s_t)
    #SLS.runAnalyticalSim(sim_rounds = 50000)

    """
    SLS.clearFrequencies()
    SLS.setFrequency(COOPERATE, 25)
    SLS.setFrequency(DEFECT, 25)
    SLS.setFrequency(PEEROSTRACIZE, 50)
    #SLS.setFrequency(POOLOSTRACIZE, 99)
    print(SLS.probaOstracizedX())
    """
