from PyTE.InfoMeasureCalculatorDiscrete import *
import numpy as np
from PyTE.utils import *

class ContextOfPastMeasureCalculatorDiscrete(InfoMeasureCalculatorDiscrete):

    def __init__(self, base, history, dontCreateObsStorage = False):
        InfoMeasureCalculatorDiscrete.__init__(self, base)
        self.k = history
        self.noObservationStorage = False
        self.nextPastCount = None #np.zeros([], dtype=int)
        self.pastCount = None #np.zeros([], dtype=int)
        self.nextCount = None #np.zeros([], dtype=int)
        self.maxShiftedValue = None #np.array([], dtype=int)
        self.base_power_k = power(self.base, self.k)

        if self.k < 0:
            raise RuntimeError("History k " + str(self.k) + " is not >= 0 for a ContextOfPAstMeasureCalculator")

        # Create constants for tracking prevValues
        self.maxShiftedValue = np.zeros(shape=self.base, dtype=int)
        for v in range(0, base):
            self.maxShiftedValue[v] = v * power(self.base, self.k-1)

        self.noObservationStorage = dontCreateObsStorage

        if not dontCreateObsStorage:
            # Create storage for counts of observations
            try:
                self.nextPastCount = np.zeros(shape=(self.base, self.base_power_k), dtype=int)
                self.pastCount = np.zeros(shape=self.base_power_k, dtype=int)
                self.nextCount = np.zeros(shape=self.base, dtype=int)
            except Exception as e:
                # Allow any Exceptions to be thrown, but catch and wrap
                # Error as a RuntimeException
                print("Requested memory for the base " + self.base +
                      " and k=" + self.k + " is too large for the JVM at this time " + str(e))

    #@classmethod
    def initialise(self):
        #InfoMeasureCalculatorDiscrete.initialise()
        #super(InfoMeasureCalculatorDiscrete, self).initialise()
        super(ContextOfPastMeasureCalculatorDiscrete, self).initialise()
        # if not self.noObservationStorage:
        #     fill(self.nextPastCount, 0)
        #     fill(self.pastCount, 0)
        #     fill(self.nextCount, 0)

    def computePastValue(self, x, t):
        pastVal = 0
        for p in range(0, self.k):
            pastVal *= self.base
            pastVal += x[t - self.k + 1 + p]

        return pastVal

    # def computePastValue(self, data, columnNumber, t):
    #     pastVal = 0
    #     for p in range(0, self.k):
    #         pastVal *= self.base
    #         pastVal += data[t - self.k + 1 + p][columnNumber]
    #
    #     return pastVal