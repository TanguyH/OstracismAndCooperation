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
ABSTAIN = "Z"
POOLRESTORE = "V'"
PEERRESTORE = "W'"

STRATEGIES = [COOPERATE, DEFECT, ABSTAIN, POOLRESTORE, PEERRESTORE]

MAX_EXP = 709.78271289338397
INFINITE_FLAG = 1350
MAX_LD = 1.18973149536e+4932

NO_RESTORATION = "no_restoration"
POOL_RESTORATION = "pool"
PEER_RESTORATION = "peer"
COMPETITION = "competition"

def comb(n, k):
    k = min(k, n-k)
    numer = reduce(op.mul, range(n, n-k, -1), 1)
    denom = reduce(op.mul, range(1, k+1), 1)
    return numer//denom

class SocialLearningSim:
    def __init__(self, N, M, c, r, G, B, gamma, beta, sigma, alpha, mu, s, compulsory = False, SOP = False, case="NO_RESTORATION"):

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

        # restoration parameters
        self.__restoration_rate = alpha

        # learning characteristics
        self.__exploration_rate = mu
        self.__imitation_strength = s

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

    def getRestorationRate(self):
        return self.__restoration_rate

    def isCompulsory(self):
        return self.__compulsory

    def SOPallowed(self):
        return self.__SOP

    def getCase(self):
        return self.__case

    def getAvailableStrategies(self):
        available_strategies = [COOPERATE, DEFECT]

        # manage Z strategy
        if not self.isCompulsory():
            available_strategies.append(ABSTAIN)

        # manage V, W strategy
        case = self.getCase()
        if case == POOL_RESTORATION:
            available_strategies.append(POOLRESTORE)
        elif case == PEER_RESTORATION:
            available_strategies.append(PEERRESTORE)
        elif case == COMPETITION:
            available_strategies.append(POOLRESTORE)
            available_strategies.append(PEERRESTORE)

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
    # SETBACK METHODS
    #---------------------------------------------------------------------------
    def valueOfInflictedHarm(self):
        M = self.getPopulationSize()
        Y, Z = self.getFrequency(DEFECT), self.getFrequency(ABSTAIN)
        harm = 0

        if Y and not (M - Z) == Y:
            harm = self.getRawPayoffY() - self.getRawPayoffX()

        return harm

    #---------------------------------------------------------------------------
    # SIM METHODS
    #---------------------------------------------------------------------------
    def getFrequencies(self):
        X = self.getFrequency(COOPERATE)
        Y = self.getFrequency(DEFECT)
        Z = self.getFrequency(ABSTAIN)
        V = self.getFrequency(POOLRESTORE)
        W = self.getFrequency(PEERRESTORE)
        return X, Y, Z, V, W

    def probaNoX(self):
        M = self.getPopulationSize()
        N = self.getSampleSize()
        X = self.getFrequency(COOPERATE)

        return comb(M - X - 2, N - 2) / comb(M - 2, N - 2)

    def probaNoXandW(self):
        M = self.getPopulationSize()
        N = self.getSampleSize()
        X = self.getFrequency(COOPERATE)
        W = self.getFrequency(PEERRESTORE)

        return comb(M - X - W - 2, N - 2) / comb(M - 2, N - 2)

    def probaNoY(self):
        M = self.getPopulationSize()
        N = self.getSampleSize()
        Y = self.getFrequency(DEFECT)

        return comb(M - Y - 2, N - 2) / comb(M - 2, N - 2)

    def probaOnlyZ(self):
        M, N = self.getPopulationSize(), self.getSampleSize()
        Z = self.getFrequency(ABSTAIN)
        return comb(Z, N - 1) / comb(M - 1, N - 1)

    def getRawPayoffX(self):
        c = self.getContribution()
        r = self.getPotFactor()
        sig = self.getEndowment()
        M, N = self.getPopulationSize(), self.getSampleSize()
        X, Y, Z, V, W = self.getFrequencies()

        # case: sole participant = dismissed
        if M - Z == 1:
            return sig

        payoff = (self.probaOnlyZ() * sig) + ((1 - self.probaOnlyZ()) * c * (r * ((M - Z - Y - 1)/(M - Z - 1)) - 1))
        return payoff

    def getPayoffX(self):
        payoff = self.getRawPayoffX()
        SOR_pool, SOR_peer = 0, 0

        X, Y, Z, V, W = self.getFrequencies()
        M, N = self.getPopulationSize(), self.getSampleSize()

        # case: sole participant = dismissed
        if M - Z == 1:
            return payoff

        # second order reimbursement
        if self.SOPallowed():
            sample_proba = (N - 1) / (M - 1)
            avg_others_in_sample = (M - X) * sample_proba

            # case pool punishment / competition
            if V:
                avg_V_in_sample = V * sample_proba
                avg_others_in_sample -= W * sample_proba                        # W helps in competition !!
                total_punitive_investment = self.getPoolFee() * avg_V_in_sample
                personal_responsibility = total_punitive_investment / (N - avg_others_in_sample)
                SOR_pool = personal_responsibility * self.getRestorationRate()

            # case peer punishment / competition
            if W:
                # punitive investment only when Y sampled
                avg_W_in_sample = W * sample_proba
                total_punitive_investment = self.getPeerFee() * avg_W_in_sample * Y * sample_proba #* (1 - self.probaNoY()) # RES 1 without proba !
                personal_responsibility = total_punitive_investment / (N - avg_others_in_sample)
                SOR_peer = personal_responsibility * self.getRestorationRate()

        return payoff - SOR_pool - SOR_peer

    def getRawPayoffY(self):
        c = self.getContribution()
        r = self.getPotFactor()
        sig = self.getEndowment()
        M, N = self.getPopulationSize(), self.getSampleSize()

        X, Y, Z, V, W = self.getFrequencies()

        # case: sole participant = dismissed
        if M - Z == 1:
            return sig

        payoff = (self.probaOnlyZ() * sig) + ((1 - self.probaOnlyZ()) * c * r * ((M - Z - Y)/(M - Z - 1)))
        return payoff

    def getPayoffY(self):
        payoff = self.getRawPayoffY()
        FOR_pool, FOR_peer = 0, 0

        M, N = self.getPopulationSize(), self.getSampleSize()
        Z = self.getFrequency(ABSTAIN)

        # case: sole participant = dismissed
        if M - Z == 1:
            return payoff

        sample_proba = (N - 1) / (M - 1)

        # case : poolrestoration / competition
        if self.getFrequency(POOLRESTORE):
            Y, V = self.getFrequency(DEFECT), self.getFrequency(POOLRESTORE)

            avg_V_in_sample =  V * sample_proba
            avg_others_in_sample =  (M - Y) * sample_proba

            total_loss_of_welfare = self.valueOfInflictedHarm() * avg_V_in_sample
            personal_responsibility = total_loss_of_welfare / (N - avg_others_in_sample)

            FOR_pool = personal_responsibility * self.getRestorationRate()

        # case : peerrestoration / competition
        if self.getFrequency(PEERRESTORE):
            Y, W = self.getFrequency(DEFECT), self.getFrequency(PEERRESTORE)

            avg_W_in_sample =  W * sample_proba
            avg_others_in_sample =  (M - Y) * sample_proba

            total_loss_of_welfare = self.valueOfInflictedHarm() * avg_W_in_sample
            personal_responsibility = total_loss_of_welfare / (N - avg_others_in_sample)

            FOR_peer = personal_responsibility * self.getRestorationRate()

        return payoff - FOR_pool - FOR_peer

    def getPayoffZ(self):
        return self.getEndowment()

    def getPayoffV(self):
        c = self.getContribution()
        r = self.getPotFactor()
        sig = self.getEndowment()
        M = self.getPopulationSize()
        G = self.getPoolFee()

        X, Y, Z, V, W = self.getFrequencies()

        assert(sig < ((r - 1) * c) - G)

        if M - Z == 1:
            return sig

        payoff = (self.probaOnlyZ() * sig) + ((1 - self.probaOnlyZ()) * (c * (r * ((M - Z - Y - 1)/(M - Z - 1)) - 1) - G))

        # APPLY RESTORATION
        restored_welfare = self.valueOfInflictedHarm() * self.getRestorationRate()

        if self.SOPallowed():
            # important : can only be restored if X or W in sample
            if self.getCase() == COMPETITION:
                restored_welfare += (G * self.getRestorationRate() * (1 - self.probaNoXandW()))
            else:
                restored_welfare += (G * self.getRestorationRate() * (1 - self.probaNoX()))

        return payoff + restored_welfare

    def getPayoffW(self):
        payoff = self.getRawPayoffX()

        M, N = self.getPopulationSize(), self.getSampleSize()
        gamma = self.getPeerFee()
        X, Y, Z, V, W = self.getFrequencies()
        sample_proba = ((N - 1) / (M - 1))

        # case: sole participant = dismissed
        if M - Z == 1:
            return self.getEndowment()

        avg_fine = gamma * sample_proba * Y
        restored_welfare = self.getRestorationRate() * self.valueOfInflictedHarm()
        restored_to_pool = 0

        if self.SOPallowed():
            # important : can only be restored if X in sample
            avg_fine += ((gamma * sample_proba * X)  * (1 - self.probaNoY()))            # need Y to see X free-rides
            restored_welfare += (avg_fine * self.getRestorationRate() * (1 - self.probaNoX()))  # need X to be reimbursed of Y

            # share to reimburse to V
            if self.getCase() == COMPETITION:
                avg_V_in_sample = V * sample_proba
                avg_others_in_sample = (M - X - W) * sample_proba

                total_punitive_investment = self.getPoolFee() * avg_V_in_sample
                personal_responsibility = total_punitive_investment / (N - avg_others_in_sample)

                restored_to_pool = personal_responsibility * self.getRestorationRate()

        return payoff - avg_fine + restored_welfare - restored_to_pool

    def getPayoff(self, strategy):
        if strategy == COOPERATE:
            return self.getPayoffX()
        elif strategy == DEFECT:
            return self.getPayoffY()
        elif strategy == ABSTAIN:
            return self.getPayoffZ()
        elif strategy == POOLRESTORE:
            return self.getPayoffV()
        else:
            return self.getPayoffW()

    def computeFermiExp(self, payoff_1, payoff_2):
        raw_exp = self.getImitationStrength() * (payoff_1 - payoff_2)
        if np.isnan(raw_exp):
            raw_exp = 0
        elif raw_exp > MAX_EXP:
            raw_exp = MAX_EXP
        return raw_exp

    #---------------------------------------------------------------------------
    #   STATIONARY DISTRIBUTIONS
    #---------------------------------------------------------------------------
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
                    t_matrix[i, j] = 1/(len(available_strategies) - 1) * fixation_proba

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

        file_name = "reimbursement/{}{}_{}.csv".format(punish_flag, name, case)

        # does not exist : create file
        file = open("{}".format(file_name), "a+")

        # write header only if file empty
        if os.stat(file_name).st_size == 0:
            header = "{}".format(factor)
            for strategy in self.getAvailableStrategies():
                header += ",{}".format(strategy)
            header += "\n"
            file.write(header)

        # write header line
        if factor == "mu":
            res_line = "{}".format(self.getExplorationRate())
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
                            alpha = .7,
                            mu = 10**-3,                # analysis done: impact clear
                            s = 0.012618568830660211,            # analysis done: impact clear
                            compulsory = False,
                            SOP = True,
                            case = PEER_RESTORATION)
    print(SLS.getAvailableStrategies())
    matrix = SLS.computeTransitionMatrix()
    print(matrix)
    s_t = SLS.computeStationaryDistribution(matrix)
    print(s_t)
    #print(SLS.valueOfInflictedHarm())

    exit()
    SLS2 = SocialLearningSim(N = 5,                      # only extreme case of N = M
                            M = 100,                    # size of M seems to impact levels
                            c = 1.,                     # seems to have impact
                            r = 3.,                     # no impact at all
                            G = .7,
                            B = .7,
                            gamma = .7,
                            beta = .7,
                            sigma = 1.,                 # only extreme case of no payoff
                            alpha = .7,
                            mu = 10**-3,                # analysis done: impact clear
                            s = np.Infinity,            # analysis done: impact clear
                            compulsory = False,
                            SOP = False,
                            case = POOL_RESTORATION)
    """
    Y_win = 0
    for i in range(1, 100):
        Y = i
        W = 100 - i

        SLS.clearFrequencies()
        SLS2.clearFrequencies()

        SLS.setFrequency(DEFECT, i)
        SLS2.setFrequency(DEFECT, i)
        SLS.setFrequency(PEERRESTORE, W)
        SLS2.setFrequency(POOLRESTORE, W)

        payoff_Y = SLS.getPayoffY()
        payoff_V = SLS.getPayoffV()
        payoff_W = SLS2.getPayoffW()

        #print("avg payoff Y : {}".format(payoff_Y))
        #print("avg payoff V' : {}".format(payoff_V))

        #if payoff_Y > payoff_V:
        #    Y_win += 1
        #    #print("Y beats V")
        #else:
        #    print(i)
        #print("--")
        print("Y: {}|V,W: {}".format(Y, W))
        print("pi(Y): {}| pi(V): {}| pi(W): {}".format(payoff_Y, payoff_V, payoff_W))

    print("Y best in {}/100".format(Y_win))
    """
