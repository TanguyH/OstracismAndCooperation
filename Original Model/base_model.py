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
POOLPUNISH = "V"
PEERPUNISH = "W"

STRATEGIES = [COOPERATE, DEFECT, ABSTAIN, POOLPUNISH, PEERPUNISH]

MAX_EXP = 709.78271289338397
INFINITE_FLAG = 1350
MAX_LD = 1.18973149536e+4932

NO_PUNISHMENT = "no_punishment"
POOL_PUNISHMENT = "pool"
PEER_PUNISHMENT = "peer"
COMPETITION = "competition"

def comb(n, k):
    k = min(k, n-k)
    numer = reduce(op.mul, range(n, n-k, -1), 1)
    denom = reduce(op.mul, range(1, k+1), 1)
    return numer//denom

class SocialLearningSim:
    def __init__(self, N, M, c, r, G, B, gamma, beta, sigma, mu, s, compulsory = False, SOP = False, case="no_punishment"):

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

        # participation status
        self.__compulsory = compulsory
        self.__SOP = SOP

        # case status
        self.__case = case

        # working vector
        self.__frequencies = [0 for strategy in STRATEGIES]

        # fixation times
        self.__ratios = {}
        self.__increase_probas = {}
        self.__fixation_probas = {}

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
        if case == POOL_PUNISHMENT:
            available_strategies.append(POOLPUNISH)
        elif case == PEER_PUNISHMENT:
            available_strategies.append(PEERPUNISH)
        elif case == COMPETITION:
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

    def clearDictionaries(self):
        self.__ratios = {}
        self.__increase_probas = {}
        self.__fixation_probas = {}

    #---------------------------------------------------------------------------
    # SIM METHODS
    #---------------------------------------------------------------------------
    def poolSetback(self):
        N = self.getSampleSize()
        M = self.getPopulationSize()
        V = self.getFrequency(POOLPUNISH)
        B = self.getPoolFine()
        return (B * (N - 1) * V) / (M - 1)

    def peerSetback(self):
        N = self.getSampleSize()
        M = self.getPopulationSize()
        W = self.getFrequency(PEERPUNISH)
        beta = self.getPeerFine()
        return (((N - 1) * W) / (M - 1)) * beta

    def getFrequencies(self):
        X = self.getFrequency(COOPERATE)
        Y = self.getFrequency(DEFECT)
        Z = self.getFrequency(ABSTAIN)
        V = self.getFrequency(POOLPUNISH)
        W = self.getFrequency(PEERPUNISH)
        return X, Y, Z, V, W

    def probaNoY(self):
        M = self.getPopulationSize()
        N = self.getSampleSize()
        Y = self.getFrequency(DEFECT)
        return comb(M - Y - 2, N - 2) / comb(M - 2, N - 2)

    def probaOnlyZ(self, Z):
        return comb(Z, self.getSampleSize() - 1) / comb(self.getPopulationSize() - 1, self.getSampleSize() - 1)

    def getRawPayoffX(self):
        c = self.getContribution()
        r = self.getPotFactor()
        sig = self.getEndowment()
        M, N = self.getPopulationSize(), self.getSampleSize()
        X, Y, Z, V, W = self.getFrequencies()

        # case: sole participant = dismissed
        if M - Z == 1:
            return sig

        payoff = (self.probaOnlyZ(Z) * sig) + ((1 - self.probaOnlyZ(Z)) * c * (r * ((M - Z - Y - 1)/(M - Z - 1)) - 1))
        return payoff

    def getPayoffX(self):
        payoff = self.getRawPayoffX()
        avg_peer_fine, avg_pool_fine = 0,0

        # adjust fines if SOP
        if self.SOPallowed():
            if self.getCase() in [POOL_PUNISHMENT, COMPETITION]:
                avg_pool_fine = self.poolSetback()
            if self.getCase() in [PEER_PUNISHMENT, COMPETITION]:
                avg_peer_fine = self.peerSetback() * (1 - self.probaNoY())

        return payoff - avg_peer_fine - avg_pool_fine

    def getPayoffY(self):
        pot_mult = self.getPotFactor() * self.getContribution()
        sig = self.getEndowment()
        M = self.getPopulationSize()
        N = self.getSampleSize()

        X, Y, Z, V, W = self.getFrequencies()

        # case: sole participant = dismissed
        if M - Z == 1:
            return sig

        payoff = (self.probaOnlyZ(Z) * sig) + ((1 - self.probaOnlyZ(Z)) * pot_mult *((M - Z - Y)/(M - Z - 1)))

        # case: allow pool punishment
        pool_setback, peer_setback = 0, 0

        if self.getCase() in [POOL_PUNISHMENT, COMPETITION]:
            pool_setback = self.poolSetback()
        if self.getCase() in [PEER_PUNISHMENT, COMPETITION]:
            peer_setback = self.peerSetback()

        #print("payoff Y : {}".format(payoff - pool_setback))

        return payoff - pool_setback - peer_setback

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

        payoff = (self.probaOnlyZ(Z) * sig) + ((1 - self.probaOnlyZ(Z)) * (c * (r * ((M - Z - Y - 1)/(M - Z - 1)) - 1) - G))
        return payoff

    def getPayoffW(self):
        N = self.getSampleSize()
        M = self.getPopulationSize()
        gamma = self.getPeerFee()
        X, Y, Z, V, W = self.getFrequencies()

        # punished by pool punishers in SOP
        pool_setback, avg_fee = 0, 0

        if self.SOPallowed():
            avg_fee = (((N - 1) * X) / (M - 1)) * gamma * (1 - self.probaNoY())
            if self.getCase() == COMPETITION:
                pool_setback = self.poolSetback()

        payoff = self.getRawPayoffX() - ((((N - 1) * Y)/(M - 1)) * gamma)
        return payoff - avg_fee - pool_setback

    def getPayoff(self, strategy):
        if strategy == COOPERATE:
            return self.getPayoffX()
        elif strategy == DEFECT:
            return self.getPayoffY()
        elif strategy == ABSTAIN:
            return self.getPayoffZ()
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
    #   FIXATION TIMES (analytical)
    #---------------------------------------------------------------------------
    def residentInvaderProba(self, resident, invader):
        M = self.getPopulationSize()
        proba_invader = self.getFrequency(invader) / M
        proba_resident = self.getFrequency(resident) / M
        return proba_resident, proba_invader

    def residentInvaderPayoff(self, resident, invader):
        return self.getPayoff(resident), self.getPayoff(invader)

    def probaIncrease(self, resident, invader, i):
        self.fixResidentInvaderFreq(resident, invader, i)
        proba_resident, proba_invader = self.residentInvaderProba(resident, invader)
        resident_payoff, invader_payoff = self.residentInvaderPayoff(resident, invader)
        return proba_invader * proba_resident * self.Fermi(invader_payoff, resident_payoff)

    def probaDecrease(self, resident, invader, i):
        self.fixResidentInvaderFreq(resident, invader, i)
        proba_resident, proba_invader = self.residentInvaderProba(resident, invader)
        resident_payoff, invader_payoff = self.residentInvaderPayoff(resident, invader)
        return proba_invader * proba_resident * self.FermiNeg(invader_payoff, resident_payoff)

    def obtainProbaIncrease(self, resident, invader, i):
        identifier = "{}{}{}".format(resident, invader, i)
        if identifier not in self.__increase_probas.keys():
            res = self.probaIncrease(resident, invader, i)
            self.__increase_probas[identifier] = res
        else:
            res = self.__increase_probas[identifier]
        return res

    def ratio(self, resident, invader, i):
        self.fixResidentInvaderFreq(resident, invader, i)
        resident_payoff, invader_payoff = self.residentInvaderPayoff(resident, invader)
        return np.exp(-self.getImitationStrength() * (invader_payoff - resident_payoff))

    def obtainRatio(self, resident, invader, i):
        identifier = "{}{}{}".format(resident, invader, i)
        if identifier not in self.__ratios.keys():
            res = self.ratio(resident, invader, i)
            self.__ratios[identifier] = res
        else:
            res = self.__ratios[identifier]
        return res

    def unconditionalFixationTime(self, resident, invader):
        fixation_proba = self.rho(resident, invader)

        sum = np.longdouble(0.0)
        for k in range(1, self.getPopulationSize()):
            sub_sum = np.longdouble(0.0)
            for l in range(1, k+1):
                frac = np.longdouble(1.)/self.obtainProbaIncrease(resident, invader, l)   # increase can be saved
                for m in range(l+1, k+1):
                    ratio = self.obtainRatio(resident, invader, m)                    # ratios can be saved
                    frac *= ratio
                sub_sum += frac
            sum += sub_sum

        return fixation_proba * sum

    def fixationProba(self, resident, invader, i):
        top = np.longdouble(1.0)
        sum = np.longdouble(0.0)
        for k in range(1, i):
            factor =np.longdouble(1.0)
            for l in range(1, k):
                factor *= self.obtainRatio(resident, invader, i)                      # ratio can be saved
            sum += factor
        top += sum

        bottom = np.longdouble(1.0)
        sum = np.longdouble(0.0)
        for k in range(1, self.getPopulationSize()):
            factor = np.longdouble(1.0)
            for l in range(1, k):
                factor *= self.obtainRatio(resident, invader, i)                      # ratio can be saved
            sum += factor
        bottom += sum

        if np.isinf(top) and np.isinf(bottom):
            res = 0.0
        else:
            res = top / bottom

        return res

    def obtainFixationProba(self, resident, invader, qty):
        identifier = "{}{}{}".format(resident, invader, qty)
        if identifier not in self.__fixation_probas.keys():
            res = self.fixationProba(resident, invader, qty)
            self.__fixation_probas[identifier] = res
        else:
            res = self.__fixation_probas[identifier]
        return res

    def conditionalFixationTime(self, resident, invader):
        sum = np.longdouble(0.0)
        for k in range(1, self.getPopulationSize()):
            sub_sum = np.longdouble(0.0)
            for l in range(1, k+1):
                fixation_proba = self.obtainFixationProba(resident, invader, l)       # can be saved
                frac = fixation_proba/self.obtainProbaIncrease(resident, invader, l)  # can be saved
                for m in range(l+1, k+1):
                    ratio = self.obtainRatio(resident, invader, m)                    # can be saved
                    frac *= ratio
                sub_sum += frac
            sum += sub_sum
        res = fixation_proba * sum
        if np.isinf(res):
            res = 0.0

        return res
    #---------------------------------------------------------------------------
    #   FIXATION TIMES (Games Revision)
    #---------------------------------------------------------------------------
    def transitionUp(self, resident, invader, invader_qty):
        M = self.getPopulationSize()
        resident_qty = M - invader_qty
        invader_payoff = self.getPayoff(invader)
        resident_payoff = self.getPayoff(resident)
        return (resident_qty / M) * (invader_qty / M) * self.Fermi(invader_payoff, resident_payoff)

    def transitionRatio(self, resident, invader):
        invader_payoff = self.getPayoff(invader)
        resident_payoff = self.getPayoff(resident)
        raw_exp = self.getImitationStrength() * (resident_payoff - invader_payoff)
        if raw_exp > MAX_EXP:
            raw_exp = MAX_EXP
        return np.exp(raw_exp)

    def directUnconditionalFixationTime(self, resident, invader):
        tau = 0.0
        fix_p = self.rho(resident, invader)
        R = 1.0
        for invader_qty in range(1, self.getPopulationSize()):
            self.fixResidentInvaderFreq(resident, invader, invader_qty)
            T_plus = self.transitionUp(resident, invader, invader_qty)
            tau = tau + (R / T_plus)
            gamma = self.transitionRatio(resident, invader)
            R = 1.0 + (gamma * R)
        return fix_p * tau

    def directFixationProbability(self, resident, invader):
        sum = 0.0
        prod = 1.0
        for invader_qty in range(1, self.getPopulationSize()):
            self.fixResidentInvaderFreq(resident, invader, invader_qty)
            gamma = self.transitionRatio(resident, invader)
            prod = prod * gamma
            sum = sum + prod

        # NOTE : test according to logic
        if(sum < 1.):
            sum = 1.

        return 1. / sum

    def directConditionalFixationTime(self, resident, invader):
        M = self.getPopulationSize()

        tau_invader = 0.0
        fix_p = self.directFixationProbability(resident, invader)
        psi = 1.0 - fix_p
        R = 1.0
        prod_gamma_list = [None for i in range(M-1)]
        prod = 1.0

        for invader_qty in range(1, M):
            self.fixResidentInvaderFreq(resident, invader, invader_qty)
            prod_gamma_list[invader_qty - 1] = prod
            gamma = self.transitionRatio(resident, invader)
            prod = prod * gamma

        for invader_qty in range(1, M):
            self.fixResidentInvaderFreq(resident, invader, M - invader_qty)
            T_plus = self.transitionUp(resident, invader, M - invader_qty)
            tau_invader = tau_invader + ((psi/T_plus) * R)
            psi = psi - (fix_p * prod_gamma_list[M - invader_qty - 1])
            gamma = self.transitionRatio(resident, invader)
            R = 1. + gamma * R

        # NOTE : test according to logic
        #if tau_invader < 0:
        #    tau_invader = 0

        return abs(tau_invader)

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

        file_name = "{}{}_{}.csv".format(punish_flag, name, case)

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
                            sigma = .1,                 # only extreme case of no payoff
                            mu = 10**-3,                # analysis done: impact clear
                            s = np.Infinity,            # analysis done: impact clear
                            compulsory = False,
                            SOP = False,
                            case = NO_PUNISHMENT)
    print(SLS.getAvailableStrategies())
    matrix = SLS.computeTransitionMatrix()
    print(matrix)
    s_t = SLS.computeStationaryDistribution(matrix)
    print(s_t)

    #print(SLS.getAvailableStrategies())
    #exit()
    #start = time.time()
    #SLS.runAnalyticalSim(50000)
    #end = time.time()
    #print(end - start)

    #print("Increase : {}".format(SLS.probaIncrease("X", "Y", 2)))
    #print("Decrease : {}".format(SLS.probaDecrease("X", "Y", 2)))
    #print("Manual ratio : {}".format(SLS.probaDecrease("X","Y", 2)/SLS.probaIncrease("X", "Y", 2)))
    #print("ratio : {}".format(SLS.ratio("X","Y", 2)))

    #print(SLS.unconditionalFixationTime("X", "Y"))
    #print(SLS.unconditionalFixationTime("Y", "X"))
    #print(np.finfo(np.longdouble).max)
    #print(SLS.conditionalFixationTime("X", "Y"))
    #print(SLS.conditionalFixationTime("Y", "X"))

    #f_time = SLS.directUnconditionalFixationTime("X", "Y")
    #f_time_2 = SLS.directConditionalFixationTime("X", "Y")
    #print("f_time = {}".format(f_time))
    #print("f_time_2 = {}".format(f_time_2))

    #print(SLS.directFixationProbability("X", "Y"))

    exit()
