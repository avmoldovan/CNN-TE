import math

class InfoMeasureCalculatorDiscrete:
    def __init__(self, base):
        self.base = base

        #Last computed average of the measure
        self.average = 0.0

        #Last computed max local value of the measure
        self.max = 0.0
        #Last computed min local value of the measure
        self.min = 0.0
        # Last computed standard deviation of local values of the measure
        self.std = 0.0
        # Number of observations supplied for the PDFs
        self.observations = 0
        # Number of available quantised states for each variable
        # (ie binary is base-2).
        # Cached value of ln(base)
        self.log_base = math.log(base)
        # Cached value of ln(2)
        self.log_2 = math.log(2.0)
        # Cache of whether the base is a power of 2
        self.power_of_2_base = False
        # Cached value of log_2(base)
        self.log_2_base = 0
        # Whether we're in debug mode
        self.debug = False

        # Construct an instance
        # @ param base number of quantisation levels for each variable.
        # E.g.binary variables are in base-2.


        log_base = 0

        if base < 2:
            raise Exception("Can't calculate info theoretic measures for base " + str(base))

        # Check if we've got a power of 2
        self.power_of_2_base = math.log(self.base, 2).is_integer();
        if self.power_of_2_base:
            self.log_2_base = round(math.log(self.base) / math.log(2))



    # Initialise the calculator for re-use with new observations.
    # (Child classes should clear the existing PDFs)
    #@classmethod
    def initialise(self):
        self.average = 0.0
        self.max = 0.0
        self.min = 0.0
        self.std = 0.0
        self.observations = 0

    # Return the measure last calculated in a call to
    # or related methods after the previous
    def getLastAverage(self):
        return self.average


    # Return the  last computed max local  value of  the measure.
    # Not declaring this final so that separable calculator
    # can throw an exception on it since it does not support it
    def getLastMax(self):
        return self.max


    # Return the last computed min local value of the measure.
    # Not declaring  this final so that separable calculator
    # can throw an exception on it since it does not support it
    def getLastMin(self):
        return self.min


    # Return the last computed standard deviation of
    # local values of the measure.
    def getLastStd(self):
        return self.std

    # Get the number of samples to be used for the PDFs here
    # which have been supplied by calls to
    # "setObservations", "addObservations" etc.
    # Note that the number of samples may not be equal to the length of time-series
    # supplied (e.g.for transfer entropy, where we need to accumulate a number of
    # samples for the past history of the destination).
    # return the number of samples to be used for the PDFs
    def getNumObservations(self) :
        return self.observations

    


    # Compute     the     average     value     of     the     measure     *      from the previously
    # supplied     samples.
    # return the     estimate     of     the     measure
    def computeAverageLocalOfObservations(self):
        pass

    
    # Set or clear     debug     mode     for extra debug printing to stdout
    def  setDebug(self, debug):
        self.debug = debug
