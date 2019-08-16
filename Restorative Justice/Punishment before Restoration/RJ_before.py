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
COOPERATE_CJ = "X_{cj}"
COOPERATE_RJ = "X_{rj}"

DEFECT_CJ = "Y_{cj}"
DEFECT_RJ = "Y_{rj}"

ABSTAIN = "Z"

POOLRESTORE = "T"
PEERRESTORE = "U"

POOLPUNISH = "V"
PEERPUNISH = "W"

STRATEGIES = [COOPERATE, COOPERATE_CJ, COOPERATE_RJ, DEFECT_CJ, DEFECT_RJ, ABSTAIN, POOLRESTORE, PEERRESTORE, POOLPUNISH, PEERPUNISH]

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
    def __init__(self, N, M, c, r, G, B, gamma, beta, sigma, alpha, psi, phi, mu, s, compulsory = False, SOP = False, case="NO_RESTORATION"):

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
        self.__redistribution_fee = psi
        self.__redistribution_satisfaction = phi

        # learning characteristics
        self.__exploration_rate = mu
        self.__imitation_strength = np.longdouble(s)

        # participation status
        self.__compulsory = compulsory
        self.__SOP = SOP

        # case status
        self.__case = case

        # working vector
        self.__frequencies = [0 for strategy in STRATEGIES]

    #---------------------------------------------------------------------------
    # ADD
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
        context = SocialLearningSim(N = self.__sample_size,
                                    M = self.__population_size,
                                    c = self.__contribution,
                                    r = self.__pot_factor,
                                    G = self.__pool_punish_fee,
                                    B = self.__pool_punish_fine,
                                    gamma = self.__peer_punish_fee,
                                    beta = self.__peer_punish_fine,
                                    sigma = self.__endowment,
                                    alpha = self.__restoration_rate,
                                    psi = self.__redistribution_fee,
                                    phi = self.__redistribution_satisfaction,
                                    mu = self.__exploration_rate,
                                    s = self.__imitation_strength,
                                    case = case,
                                    compulsory = self.__compulsory,
                                    SOP = self.__SOP)
        available_strategies = context.getAvailableStrategies()

        # file name formatting
        name = self.generateFileName(case)
        data_file = open("data/{}/{}.csv".format(factor, name), "w")

        # format header
        header = "{}".format(factor)
        for strategy in available_strategies:
            header += ",{}".format(strategy)
        data_file.write("{}\n".format(header))

        return data_file

    #---------------------------------------------------------------------------
    #   WRITER
    #---------------------------------------------------------------------------
    def writeCalcAndWrite(self, sim, factor, file):
        trans = sim.computeTransitionMatrix()
        stationary = sim.computeStationaryDistribution(trans)

        res = "{}".format(factor)
        for distribution in stationary:
            res += ",{}".format(distribution)
        res += "\n"

        file.write(res)
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

    def getRedistributionFee(self):
        return self.__redistribution_fee

    def getRedistributionSatisfaction(self):
        return self.__redistribution_satisfaction

    def isCompulsory(self):
        return self.__compulsory

    def SOPallowed(self):
        return self.__SOP

    def getCase(self):
        return self.__case

    def getAvailableStrategies(self):
        if self.SOPallowed():
            available_strategies = [COOPERATE_CJ, COOPERATE_RJ]
        else:
            available_strategies = [COOPERATE]

        available_strategies.append(DEFECT_CJ)
        available_strategies.append(DEFECT_RJ)

        # manage Z strategy
        if not self.isCompulsory():
            available_strategies.append(ABSTAIN)

        # manage V, W strategy
        case = self.getCase()
        if case == POOL_RESTORATION:
            available_strategies.append(POOLRESTORE)
            available_strategies.append(POOLPUNISH)
        elif case == PEER_RESTORATION:
            available_strategies.append(PEERRESTORE)
            available_strategies.append(PEERPUNISH)
        elif case == COMPETITION:
            available_strategies.append(POOLRESTORE)
            available_strategies.append(PEERRESTORE)
            available_strategies.append(POOLPUNISH)
            available_strategies.append(PEERPUNISH)

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
    def poolSetback(self):
        N = self.getSampleSize()
        M = self.getPopulationSize()
        T = self.getFrequency(POOLRESTORE)
        V = self.getFrequency(POOLPUNISH)
        B = self.getPoolFine()
        return (B * (N - 1) * (T + V)) / (M - 1)

    def peerSetback(self):
        N = self.getSampleSize()
        M = self.getPopulationSize()
        U = self.getFrequency(PEERRESTORE)
        W = self.getFrequency(PEERPUNISH)
        beta = self.getPeerFine()
        return (beta * (N - 1) * (U + W)) / (M - 1)

    def valueOfInflictedHarm(self):
        Y = self.getFrequencyY()
        harm = 0
        if Y:
            harm = self.getRawPayoffY() - self.getRawPayoffX()
        return harm

    #---------------------------------------------------------------------------
    # SIM METHODS
    #---------------------------------------------------------------------------
    def getFrequencies(self):
        X = self.getFrequencyX()
        Y = self.getFrequencyY()
        Z = self.getFrequency(ABSTAIN)
        T = self.getFrequency(POOLRESTORE)
        U = self.getFrequency(PEERRESTORE)
        V = self.getFrequency(POOLPUNISH)
        W = self.getFrequency(PEERPUNISH)
        return X, Y, Z, T, U, V, W

    def getFrequencyX(self):
        X = self.getFrequency(COOPERATE)
        if self.SOPallowed():
            X = self.getFrequency(COOPERATE_CJ) + self.getFrequency(COOPERATE_RJ)
        return X

    def getFrequencyY(self):
        return self.getFrequency(DEFECT_CJ) + self.getFrequency(DEFECT_RJ)

    def probaNoY(self):
        M = self.getPopulationSize()
        N = self.getSampleSize()
        Y = self.getFrequencyY()

        return comb(M - Y - 2, N - 2) / comb(M - 2, N - 2)

    def probaNoYrj(self):
        M = self.getPopulationSize()
        N = self.getSampleSize()
        Y_rj = self.getFrequency(DEFECT_RJ)

        return comb(M - Y_rj - 2, N - 2) / comb(M - 2, N - 2)

    def probaNoYcj(self):
        M = self.getPopulationSize()
        N = self.getSampleSize()
        Y_cj = self.getFrequency(DEFECT_CJ)

        return comb(M - Y_cj - 2, N - 2) / comb(M - 2, N - 2)

    def probaOnlyZ(self):
        M, N = self.getPopulationSize(), self.getSampleSize()
        Z = self.getFrequency(ABSTAIN)
        return comb(Z, N - 1) / comb(M - 1, N - 1)

    def getRawPayoffX(self):
        c = self.getContribution()
        r = self.getPotFactor()
        sig = self.getEndowment()
        M, N = self.getPopulationSize(), self.getSampleSize()
        Z, Y = self.getFrequency(ABSTAIN), self.getFrequencyY()

        # case: sole participant = dismissed
        if M - Z == 1:
            return sig

        payoff = (self.probaOnlyZ() * sig) + ((1 - self.probaOnlyZ()) * c * (r * ((M - Z - Y - 1)/(M - Z - 1)) - 1))
        return payoff

    def getPayoffXcj(self):
        payoff = self.getRawPayoffX()
        avg_peer_fine, avg_pool_fine = 0,0

        M, N = self.getPopulationSize(), self.getSampleSize()
        Z = self.getFrequency(ABSTAIN)

        if M - Z == 1:
            return self.getEndowment()

        # adjust fines if SOP
        if self.SOPallowed():
            if self.getCase() in [POOL_RESTORATION, COMPETITION]:
                avg_pool_fine = self.poolSetback()
            if self.getCase() in [PEER_RESTORATION, COMPETITION]:
                avg_peer_fine = self.peerSetback() * (1 - self.probaNoY())

        return payoff - avg_peer_fine - avg_pool_fine


    def getPayoffXrj(self):
        payoff = self.getRawPayoffX()

        SOP_pool_restored, SOP_peer_restored = 0, 0
        avg_pool_fine, avg_peer_fine = 0, 0

        M, N = self.getPopulationSize(), self.getSampleSize()
        Z = self.getFrequency(ABSTAIN)

        if M - Z == 1:
            return self.getEndowment()

        if self.SOPallowed():

            # punishment before ..
            if self.getCase() in [POOL_RESTORATION, COMPETITION]:
                avg_pool_fine = self.poolSetback()
            if self.getCase() in [PEER_RESTORATION, COMPETITION]:
                avg_peer_fine = self.peerSetback() * (1 - self.probaNoY())

            # .. restoration
            sample_proba = (N - 1) / (M - 1)
            X_rj, T, U = self.getFrequency(COOPERATE_RJ), self.getFrequency(POOLRESTORE), self.getFrequency(PEERRESTORE)
            # case : pool restoration / competition
            personal_welfare_restoration = self.valueOfInflictedHarm() * self.getRestorationRate() * (1 - self.probaNoYrj()) #* sample_proba
            if T:
                # welfare restored in sample
                avg_others_in_sample = (M - X_rj - T) * sample_proba
                total_restored_welfare_loss = personal_welfare_restoration * T * sample_proba
                SOP_pool_restored = total_restored_welfare_loss / (N - avg_others_in_sample)

            # case : peer restoration / competition
            if U:
                avg_others_in_sample = (M - X_rj - U) * sample_proba
                total_restored_welfare_loss = personal_welfare_restoration * U * sample_proba
                SOP_peer_restored = total_restored_welfare_loss / (N - avg_others_in_sample)

        return payoff - avg_pool_fine - avg_peer_fine + SOP_pool_restored + SOP_peer_restored

    def getRawPayoffY(self):
        c = self.getContribution()
        r = self.getPotFactor()
        sig = self.getEndowment()
        M, N = self.getPopulationSize(), self.getSampleSize()
        Z, Y = self.getFrequency(ABSTAIN), self.getFrequencyY()

        # case: sole participant = dismissed
        if M - Z == 1:
            return sig

        payoff = (self.probaOnlyZ() * sig) + ((1 - self.probaOnlyZ()) * c * r * ((M - Z - Y)/(M - Z - 1)))
        return payoff

    def getPayoffYcj(self):
        payoff = self.getRawPayoffY()
        M = self.getPopulationSize()
        Z = self.getFrequency(ABSTAIN)

        if M - Z == 1:
            return self.getEndowment()

        pool_setback, peer_setback = 0, 0

        if self.getCase() in [POOL_RESTORATION, COMPETITION]:
            pool_setback = self.poolSetback()
        if self.getCase() in [PEER_RESTORATION, COMPETITION]:
            peer_setback = self.peerSetback()

        return payoff - pool_setback - peer_setback

    def getPayoffYrj(self):
        payoff = self.getRawPayoffY()
        FOR_pool, FOR_peer = 0, 0
        pool_setback, peer_setback = 0, 0

        M, N = self.getPopulationSize(), self.getSampleSize()
        Z = self.getFrequency(ABSTAIN)

        if M - Z == 1:
            return self.getEndowment()

        sample_proba = (N - 1) / (M - 1)

        # case : poolrestoration / competition
        if self.getCase() in [POOL_RESTORATION, COMPETITION]:
            # punishment before ..
            pool_setback = self.poolSetback()

            # .. restoration
            Y_rj, T = self.getFrequency(DEFECT_RJ), self.getFrequency(POOLRESTORE)

            avg_T_in_sample =  T * sample_proba
            avg_others_in_sample =  (M - Y_rj) * sample_proba

            total_loss_of_welfare = self.valueOfInflictedHarm() * avg_T_in_sample
            personal_responsibility = total_loss_of_welfare / (N - avg_others_in_sample)

            FOR_pool = personal_responsibility * self.getRestorationRate()

        # case : peerrestoration / competition
        if self.getCase() in [PEER_RESTORATION, COMPETITION]:
            # punishment before ..
            peer_setback = self.peerSetback()

            # .. restoration
            Y_rj, U = self.getFrequency(DEFECT_RJ), self.getFrequency(PEERRESTORE)

            avg_U_in_sample =  U * sample_proba
            avg_others_in_sample =  (M - Y_rj) * sample_proba

            total_loss_of_welfare = self.valueOfInflictedHarm() * avg_U_in_sample
            personal_responsibility = total_loss_of_welfare / (N - avg_others_in_sample)

            FOR_peer = personal_responsibility * self.getRestorationRate()

        return payoff - pool_setback - peer_setback - FOR_pool - FOR_peer

    def getPayoffZ(self):
        return self.getEndowment()

    def getPayoffT(self):
        c = self.getContribution()
        r = self.getPotFactor()
        sig = self.getEndowment()
        M, N = self.getPopulationSize(), self.getSampleSize()
        G = self.getPoolFee()

        Y, Z = self.getFrequencyY(), self.getFrequency(ABSTAIN)

        assert(sig < ((r - 1) * c) - G)

        if M - Z == 1:
            return sig

        payoff = (self.probaOnlyZ() * sig) + ((1 - self.probaOnlyZ()) * (c * (r * ((M - Z - Y - 1)/(M - Z - 1)) - 1) - G))

        # APPLY RESTORATION
        restored_welfare = self.valueOfInflictedHarm() * self.getRestorationRate() * (1 - self.probaNoYrj())
        redistributed_restoration, satisfaction = 0, 0

        if self.SOPallowed():
            X_rj = self.getFrequency(COOPERATE_RJ)

            sample_proba = (N - 1) / (M - 1)
            avg_X_in_sample = sample_proba * X_rj

            satisfaction = self.getRedistributionSatisfaction() * avg_X_in_sample * (1 - self.probaNoYrj())
            redistributed_restoration = (restored_welfare / N) * avg_X_in_sample * (1 - self.probaNoYrj())

        return payoff + restored_welfare - redistributed_restoration + satisfaction

    def getPayoffU(self):
        payoff = self.getRawPayoffX()

        M, N = self.getPopulationSize(), self.getSampleSize()
        gamma = self.getPeerFee()
        Y, Z = self.getFrequencyY(), self.getFrequency(ABSTAIN)
        pool_setback = 0

        if M - Z == 1:
            return self.getEndowment()

        sample_proba = (N - 1) / (M - 1)

        avg_fine = gamma * sample_proba * Y
        restored_welfare = self.getRestorationRate() * self.valueOfInflictedHarm() * (1 - self.probaNoYrj())
        redistributed_restoration, satisfaction = 0, 0

        if self.SOPallowed():
            # redistribution
            X_rj = self.getFrequency(COOPERATE_RJ)
            avg_X_in_sample = sample_proba * X_rj
            avg_fine += (self.getRedistributionFee() * avg_X_in_sample * (1 - self.probaNoYrj()))

            # punishment
            avg_fine += self.getPeerFee() * self.getFrequencyX() * sample_proba * (1 - self.probaNoY())

            satisfaction = self.getRedistributionSatisfaction() * avg_X_in_sample * (1 - self.probaNoYrj())
            redistributed_restoration = (restored_welfare / N) * avg_X_in_sample * (1 - self.probaNoYrj())

            if self.getCase() == COMPETITION:
                pool_setback = self.poolSetback()

        return payoff - avg_fine + restored_welfare - redistributed_restoration + satisfaction - pool_setback

    def getPayoffV(self):
        c = self.getContribution()
        r = self.getPotFactor()
        sig = self.getEndowment()
        M = self.getPopulationSize()
        G = self.getPoolFee()

        Y, Z = self.getFrequencyY(), self.getFrequency(ABSTAIN)

        assert(sig < ((r - 1) * c) - G)

        if M - Z == 1:
            return sig

        payoff = (self.probaOnlyZ() * sig) + ((1 - self.probaOnlyZ()) * (c * (r * ((M - Z - Y - 1)/(M - Z - 1)) - 1) - G))
        return payoff

    def getPayoffW(self):
        N = self.getSampleSize()
        M = self.getPopulationSize()
        gamma = self.getPeerFee()
        Z = self.getFrequency(ABSTAIN)

        if M - Z == 1:
            return self.getEndowment()

        # punished by pool punishers in SOP
        pool_setback, avg_fee = 0, 0
        Y = self.getFrequencyY()

        if self.SOPallowed():
            X = self.getFrequencyX()

            avg_fee = ((gamma * (N - 1) * X) / (M - 1)) * (1 - self.probaNoY())
            if self.getCase() == COMPETITION:
                pool_setback = self.poolSetback()

        # reduce payoff by punishment cost
        payoff = self.getRawPayoffX() - ((gamma * (N - 1) * Y)/(M - 1))

        return payoff - avg_fee - pool_setback

    def getPayoff(self, strategy):
        if strategy == COOPERATE:
            return self.getRawPayoffX()
        elif strategy == COOPERATE_CJ:
            return self.getPayoffXcj()
        elif strategy == COOPERATE_RJ:
            return self.getPayoffXrj()
        elif strategy == DEFECT_CJ:
            return self.getPayoffYcj()
        elif strategy == DEFECT_RJ:
            return self.getPayoffYrj()
        elif strategy == ABSTAIN:
            return self.getPayoffZ()
        elif strategy == POOLRESTORE:
            return self.getPayoffT()
        elif strategy == PEERRESTORE:
            return self.getPayoffU()
        elif strategy == POOLPUNISH:
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

        file_name = "RJbefore/{}{}_{}.csv".format(punish_flag, name, case)

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
                            psi = 0.,                   # redistribution fee
                            phi = 0.,                   # satisfaction -> influences results .4 !!
                            mu = 10**-3,                # analysis done: impact clear
                            s = np.Infinity,            # analysis done: impact clear
                            compulsory = True,
                            SOP = True,
                            case = COMPETITION)
    print(SLS.getAvailableStrategies())
    matrix = SLS.computeTransitionMatrix()
    print(matrix)
    s_t = SLS.computeStationaryDistribution(matrix)
    print(s_t)
