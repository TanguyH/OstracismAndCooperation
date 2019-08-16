import numpy as np

class Agent:
    def __init__(self, available_strategies):
        self.__available_strategies = available_strategies
        self.__payoff = 0.0

        self.assignStrategy()

    #---------------------------------------------------------------------------
    # GETTERS
    #---------------------------------------------------------------------------
    def getAvailableStrategies(self, i = None):
        if i == None:
            res = self.__available_strategies
        else:
            res = self.__available_strategies[i]
        return res

    def getStrategy(self):
        return self.__strategy

    def getPayoff(self):
        return self.__payoff

    #---------------------------------------------------------------------------
    # SETTERS
    #---------------------------------------------------------------------------
    def setStrategy(self, new_strategy):
        self.__strategy = new_strategy

    def setPayoff(self, new_payoff):
        self.__payoff = new_payoff

    #---------------------------------------------------------------------------
    # INIT METHODS
    #---------------------------------------------------------------------------
    def assignStrategy(self):
        proba_per_strategy = 1. / len(self.getAvailableStrategies())
        proba = np.random.random()
        strategy_index = int(proba // proba_per_strategy)
        self.__strategy = self.getAvailableStrategies(strategy_index)

    #---------------------------------------------------------------------------
    # SIM METHODS
    #---------------------------------------------------------------------------
    def switchToOtherAvailableStrategy(self):
        available_strategies = self.getAvailableStrategies()

        proba_per_strategy = 1. / len(available_strategies)
        proba = np.random.random()

        my_strategy_index = available_strategies.index(self.getStrategy())
        new_strategy_index = int(proba // proba_per_strategy)

        while(my_strategy_index == new_strategy_index):
            proba = np.random.random()
            new_strategy_index = int(proba // proba_per_strategy)
        self.setStrategy(available_strategies[new_strategy_index])

if __name__ == "__main__":
    agent = Agent("no_punishment", ["Y", "X"])
    print(agent.getStrategy())
    agent.switchToAvailableStrategy()
