import numpy as np

from PyTE.ContextOfPastMeasureCalculatorDiscrete import *
from array import *
import torch

class TEWindow:

    def __init__(self,
                 clean_window = True,
                 MA_window = 0,
                 base = 2,
                 destHistoryEmbedLength = 1,
                 destEmbeddingDelay = 1,
                 sourceHistoryEmbeddingLength = 1,
                 sourceEmbeddingDelay = 1,
                 delay = 1,
                 history = 1):
        #ContextOfPastMeasureCalculatorDiscrete.__init__(self, base, destHistoryEmbedLength)
        self.base_power_l = power(base, sourceHistoryEmbeddingLength)

        self.init(MA_window, base, destHistoryEmbedLength, destEmbeddingDelay, sourceHistoryEmbeddingLength, sourceEmbeddingDelay, delay, history)
        self.tes = np.array([])  # torch.empty(size=(self.MA_window,0), dtype=torch.float32, device='cpu', requires_grad=False)
        self.clean_window = clean_window

    def init(self,
             MA_window = 0,
             base = 2,
             destHistoryEmbedLength = 1,
             destEmbeddingDelay = 1,
             sourceHistoryEmbeddingLength = 1,
             sourceEmbeddingDelay = 1,
             delay = 1,
             dontCreateObsStorage = False):

        self.base = base

        # Last computed average of the measure
        self.average = 0.0

        # Last computed max local value of the measure
        self.max = 0.0
        # Last computed min local value of the measure
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

        self.k = destHistoryEmbedLength

        self.base_power_k = power(self.base, self.k)

        if self.k < 0:
            raise RuntimeError("destHistoryEmbedLength k " + str(self.k) + " is not >= 0 for a TEWindow")

        self.maxShiftedValue = np.ndarray(shape=self.base, dtype=int)
        for v in range(0, base):
            self.maxShiftedValue[v] = v * power(self.base, self.k - 1)

        self.noObservationStorage = dontCreateObsStorage

        if not dontCreateObsStorage:
            # Create storage for counts of observations
            try:
                self.nextPastCount = np.zeros(shape=(self.base, self.base_power_k), dtype=int)
                self.pastCount = np.zeros(shape=self.base_power_k, dtype=int)
                #self.nextCount = np.zeros(shape=self.base, dtype=int)
            except Exception as e:
                # Allow any Exceptions to be thrown, but catch and wrap
                # Error as a RuntimeException
                print("Requested memory for the base " + self.base +
                      " and k=" + self.k + " is too large for the JVM at this time " + str(e))

        self.sourceNextPastCount = np.ndarray(shape=(self.base_power_l, self.base, self.base_power_k), dtype=int)
        self.pastCount = np.zeros(shape=self.base_power_k, dtype=int)
        self.nextPastCount = np.zeros(shape=(self.base, self.base_power_k), dtype=int)
        self.sourcePastCount = np.ndarray(shape=(self.base_power_l, self.base_power_k), dtype=int)
        self.periodicBoundaryConditions = True
        self.base = 2
        self.startObservationTime = 1
        self.destEmbeddingDelay = destEmbeddingDelay
        if sourceHistoryEmbeddingLength <= 0:
            raise RuntimeError("Cannot have source embedding length of zero or less")
        self.sourceHistoryEmbedLength = sourceHistoryEmbeddingLength
        self.sourceEmbeddingDelay = sourceEmbeddingDelay
        self.delay = delay
        self.prev_te = None

        ##		// Check that we can convert the history value into an integer ok:
        ## if (sourceHistoryEmbedLength > Math.log(Integer.MAX_VALUE) / log_base) {
        ##	throw new RuntimeException("Base and source history combination too large");
        ##}

        self.maxShiftedSourceValue = np.ndarray(shape=base, dtype=int)
        for v in range(0, base):
            self.maxShiftedSourceValue[v] = v * power(base, self.sourceHistoryEmbedLength - 1)

        # Create storage for extra counts of observations
        # self.sourceNextPastCount = np.ndarray(shape=(self.base_power_l, base, self.base_power_k), dtype=int)
        # self.sourcePastCount = np.ndarray(shape=(self.base_power_l, self.base_power_k), dtype=int)
        self.sourceNextPastCount = np.zeros(shape=(self.base_power_l, base, self.base_power_k), dtype=int)
        self.sourcePastCount = np.zeros(shape=(self.base_power_l, self.base_power_k), dtype=int)

        # Which time step do we start taking observations from?
        # These two integers represent the earliest next time step, in the cases where the destination
        # embedding itself determines where we can start taking observations, or
        # the case where the source embedding plus delay is longer and so determines
        # where we can start taking observations.
        startTimeBasedOnDestPast = (self.k - 1) * destEmbeddingDelay + 1;
        startTimeBasedOnSourcePast = (self.sourceHistoryEmbedLength - 1) * sourceEmbeddingDelay + delay;

        self.startObservationTime = max(startTimeBasedOnDestPast, startTimeBasedOnSourcePast)
        self.estimateComputed = False

        # self.xs, self.xs_ = torch.empty(size=(0,), dtype=torch.uint8, device='cpu'), torch.empty(size=(0,), dtype=torch.uint8, device='cpu')  # array('h'), array('h')
        # self.ys, self.ys_ = torch.empty(size=(0,), dtype=torch.uint8, device='cpu'), torch.empty(size=(0,), dtype=torch.uint8, device='cpu'),  # array('h'), array('h')
        self.xs, self.xs_ = torch.empty(size=(0,), dtype=torch.uint8, device='cpu'), torch.empty(size=(0,), dtype=torch.uint8, device='cpu')  # array('h'), array('h')
        self.ys, self.ys_ = torch.empty(size=(0,), dtype=torch.uint8, device='cpu'), torch.empty(size=(0,), dtype=torch.uint8, device='cpu'),  # array('h'), array('h')
        self.MA_window = MA_window



    def initialise(self):
        #ContextOfPastMeasureCalculatorDiscrete.initialise()
        #super(ContextOfPastMeasureCalculatorDiscrete, self).initialise()

        #super(TEWindow, self).initialise()
        self.average = 0.0
        self.max = 0.0
        self.min = 0.0
        self.std = 0.0
        self.observations = 0

        self.estimateComputed = False

        fill(self.sourceNextPastCount, 0)
        fill(self.sourcePastCount, 0)

        self.xs, self.xs_ = torch.empty(size=(0,),dtype=torch.uint8, device='cpu'), torch.empty(size=(0,),dtype=torch.uint8, device='cpu', requires_grad=False) #array('h'), array('h')
        self.ys, self.ys_ = torch.empty(size=(0,),dtype=torch.uint8, device='cpu'), torch.empty(size=(0,),dtype=torch.uint8, device='cpu', requires_grad=False), #array('h'), array('h')


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


    def computePastValue(self, x, t):
        pastVal = 0
        for p in range(0, self.k):
            pastVal *= self.base
            pastVal += x[t - self.k + 1 + p]

        return pastVal


    def __len__(self):
        if(self.xs != None):
            return len(self.xs)
        return 0

    def add_source(self, src):
        if self.clean_window:
            self.init(self.MA_window)
        self._add_item(True, src)

    def add_dest(self, dst):
        #this cleanup is dependent on the previous step (add_source()) to do it
        #self.init(self.MA_window)
        self._add_item(False, dst)

    def _add_item(self, src_or_dest, item: torch.Tensor):
        if src_or_dest:
            self.xs_ = item #np.concatenate((self.xs_, item))
            # self.xs_ = item
        else:
            self.ys_ = item #np.concatenate((self.ys_, item))

        # if src_or_dest:
        #     self.xs_ = item
        #     # self.xs_ = item
        # else:
        #     self.ys_ = item


        # if item is not None and len(item) > 1:
        #     if src_or_dest:
        #         self.xs_ = np.concatenate((self.xs_, item))
        #         #self.xs_ = item
        #     else:
        #         self.ys_ = np.concatenate((self.ys_, item))
        #         #self.ys_ = item
        # elif item is not None:
        #     if src_or_dest:
        #         self.xs_ = torch.cat((self.xs_, torch.tensor(item)))
        #     else:
        #         self.ys_ = torch.cat((self.ys_, torch.tensor(item)))

    def addOnlineObservationsLag1(self, source, dest, startTime=0, endTime=0):
        if endTime == 0:
            endTime = len(dest) - 1

        if ((endTime - startTime) <= 0):
            # No observations to add
            return

        if (endTime >= len(dest) or endTime >= len(source)):
            msg = "endTime {:d} must be <= length of input arrays (dest: {:d}, source: {:d})".format(endTime,
                                                                                                     dest.shape[0],
                                                                                                     source.shape[0])
            raise RuntimeError(msg)

        self.observations += (endTime - startTime)

        # Initialise and store the current previous values;
        #  one for each phase of the embedding delay.
        # First for the destination:
        pastVal = np.ndarray(shape=1, dtype=int)
        pastVal[0] = 0

        sourcePastVal = np.ndarray(shape=self.sourceEmbeddingDelay, dtype=int)
        sourcePastVal[0] = 0

        destVal = 0
        startIndex = startTime + 1
        endIndex = endTime + 1
        for r in list(range(startIndex, endIndex)):
            if self.k > 0:
                pastVal[0] += dest[r - 1]
            sourcePastVal[0] += source[r - self.delay]
            # Add to the count for this particular transition
            # (cell's assigned as above
            destVal = dest[r]
            thisPastVal = pastVal[0]
            thisSourceVal = sourcePastVal[0]
            self.sourceNextPastCount[thisSourceVal][destVal][thisPastVal] += 1
            self.sourcePastCount[thisSourceVal][thisPastVal] += 1
            self.nextPastCount[destVal][thisPastVal] += 1
            self.pastCount[thisPastVal] += 1
            #self.nextCount[destVal] += 1
            # Now, update the combined embedding values and phases,
            # for this phase we back out the oldest value which we'll no longer need:
            if self.k > 0:
                pastVal[0] -= self.maxShiftedValue[dest[r - 1 - (self.k - 1)]]
                pastVal[0] *= self.base

            sourcePastVal[0] -= self.maxShiftedSourceValue[source[r - self.delay - (self.sourceHistoryEmbedLength - 1) * self.sourceEmbeddingDelay]]
            sourcePastVal[0] *= self.base
            # then update the phase
            #destEmbeddingPhase = (destEmbeddingPhase + 1) % self.destEmbeddingDelay
            #sourceEmbeddingPhase = (sourceEmbeddingPhase + 1) % self.sourceEmbeddingDelay

    # def addOnlineObservationsLag1(self, source, dest, startTime=0, endTime=0):
    #     if endTime == 0:
    #         endTime = len(dest) - 1
    #
    #     if ((endTime - startTime) - self.startObservationTime + 1 <= 0):
    #         # No observations to add
    #         return
    #
    #     if (endTime >= len(dest) or endTime >= len(source)):
    #         msg = "endTime {:d} must be <= length of input arrays (dest: {:d}, source: {:d})".format(endTime,
    #                                                                                                  dest.shape[0],
    #                                                                                                  source.shape[0])
    #         raise RuntimeError(msg)
    #
    #     self.observations += (endTime - startTime) - self.startObservationTime + 1
    #
    #     # Initialise and store the current previous values;
    #     #  one for each phase of the embedding delay.
    #     # First for the destination:
    #     pastVal = np.ndarray(shape=self.destEmbeddingDelay, dtype=int)
    #     pastVal[0] = 0
    #
    #     sourcePastVal = np.ndarray(shape=self.sourceEmbeddingDelay, dtype=int)
    #     sourcePastVal[0] = 0
    #
    #     destVal = 0
    #     destEmbeddingPhase = 0
    #     sourceEmbeddingPhase = 0
    #     startIndex = startTime + self.startObservationTime
    #     endIndex = endTime + 1
    #     for r in list(range(startIndex, endIndex)):
    #         if self.k > 0:
    #             pastVal[destEmbeddingPhase] += dest[r - 1]
    #         sourcePastVal[sourceEmbeddingPhase] += source[r - self.delay]
    #         # Add to the count for this particular transition
    #         # (cell's assigned as above
    #         destVal = dest[r]
    #         thisPastVal = pastVal[destEmbeddingPhase]
    #         thisSourceVal = sourcePastVal[sourceEmbeddingPhase]
    #         self.sourceNextPastCount[thisSourceVal][destVal][thisPastVal] += 1
    #         self.sourcePastCount[thisSourceVal][thisPastVal] += 1
    #         self.nextPastCount[destVal][thisPastVal] += 1
    #         self.pastCount[thisPastVal] += 1
    #         self.nextCount[destVal] += 1
    #         # Now, update the combined embedding values and phases,
    #         # for this phase we back out the oldest value which we'll no longer need:
    #         if self.k > 0:
    #             pastVal[destEmbeddingPhase] -= self.maxShiftedValue[dest[r - 1 - (self.k - 1) * self.destEmbeddingDelay]]
    #             pastVal[destEmbeddingPhase] *= self.base
    #
    #         sourcePastVal[sourceEmbeddingPhase] -= self.maxShiftedSourceValue[source[r - self.delay - (self.sourceHistoryEmbedLength - 1) * self.sourceEmbeddingDelay]]
    #         sourcePastVal[sourceEmbeddingPhase] *= self.base
    #         # then update the phase
    #         #destEmbeddingPhase = (destEmbeddingPhase + 1) % self.destEmbeddingDelay
    #         #sourceEmbeddingPhase = (sourceEmbeddingPhase + 1) % self.sourceEmbeddingDelay


    def calcLocalTE(self):
        if (np.all(self.xs_ == 0) and np.all(self.ys_ == 0)):
            temp_te = 0.
        else:
            self.addOnlineObservationsLag1(self.xs_, self.ys_)
            temp_te = self.computeAverageLocalOfObservations()

        self.xs = np.concatenate((self.xs, self.xs_))
        self.ys = np.concatenate((self.ys, self.ys_))

        self.tes = np.append(self.tes, temp_te)
        # in order not to keep the source timeseries in both locations
        # we are continuously appending here these, while in the netwowrk these are removed


        #since we use a SMA and we maintain the previous TE value there is no need to keep the historical series
        # self.xs = torch.cat((self.xs[1:], self.xs_.cpu()), dim=0)
        # self.ys = torch.cat((self.ys[1:], self.ys_.cpu()), dim=0)

        #self.prev_te = self.prev_te + temp_te
        #return self.prev_te / 2

        #return (self.tes[self.MA_window+1:].sum() + temp_te) / self.MA_window

        #res = (self.tes[-1]+self.tes[-2])/2

        if (len(self.tes) > 1):
            res = self.tes[-2] - self.tes[-1] #temp_te
        else:
            res = temp_te

        if res < 1.0e-8:
            return 0.
        else:
            if res > 10.:
                return 1.0
            else:
                return res

    def computeAverageLocalOfObservations(self):
        te = 0.0
        teCont = 0.0

        self.max = 0
        self.min = 0
        meanSqLocals = 0.0

        for pastVal in range(0, self.base_power_k):
            # compute p(past)
            # double p_past = (double) pastCount[pastVal] / (double) observations
            if self.pastCount[pastVal] == 0:
                continue

            for destVal in range(0, self.base):
                # compute p(dest, past)
                # double p_dest_past = (double) destPastCount[destVal][pastVal] / (double) observations;
                if self.nextPastCount[destVal][pastVal] == 0:
                    continue

                denom = float(self.nextPastCount[destVal][pastVal]) / float(self.pastCount[pastVal])
                for sourceVal in range(0, self.base_power_l):
                    if self.sourceNextPastCount[sourceVal][destVal][pastVal] != 0:
                        # compute p(source, dest, past)
                        p_source_dest_past = float(self.sourceNextPastCount[sourceVal][destVal][pastVal]) / float(self.observations)
                        #print("observations: " + str(self.observations) + " base_power_l: " + str(self.base_power_l) + " sourceHistoryEmbedLength: " + str(self.sourceHistoryEmbedLength))
                        #print(" xn: " + str(pastVal) + " yn: " + str(sourceVal) + " xn+1: " + str(destVal))

                        # pinin1 = (float(pastVal) / float(self.observations))*(float(sourceVal) / float(self.observations))
                        # pinjn = (float(pastVal) / float(self.observations)) * (float(sourceVal) / float(self.observations))

                        logTerm = (float(self.sourceNextPastCount[sourceVal][destVal][pastVal]) / float(self.sourcePastCount[sourceVal][pastVal]) / denom)
                        localValue = math.log(logTerm)
                        #print("demon: " + str(denom) + " logTerm: " + str(logTerm) + " localValu: " + str(localValue))
                        #print("p(in, in1, jn): " + str(p_source_dest_past) +
                        #      " p(in, jn): " + str(float(self.sourcePastCount[sourceVal][pastVal]) / float(self.observations)) +
                        #      " p(in, in1): " + str(float(self.nextPastCount[destVal][pastVal]) / float(self.observations)) +
                        #      " p(in): " + str(float(self.pastCount[pastVal]) / float(self.observations)))

                        teCont = p_source_dest_past * localValue
                        if localValue > self.max:
                            self.max = localValue
                        elif localValue < self.min:
                            self.min = localValue

                        # add this contribution to the mean
                        # of the squared local values
                        meanSqLocals += teCont * localValue
                        # if teCont > 0.0:
                        #     print("teCont: " + str(teCont))
                        #     print("---")
                    else:
                        teCont = .0

                    te += teCont
        te = te / self.log_2
        self.max = self.max / self.log_2
        self.min = self.min / self.log_2
        meanSqLocals = meanSqLocals / (self.log_2 * self.log_2)

        self.average = te
        #self.std = math.sqrt(meanSqLocals - self.average * self.average)
        self.estimateComputed = True
        self.sum = te
        return te