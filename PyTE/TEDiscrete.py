from PyTE.ContextOfPastMeasureCalculatorDiscrete import *
from array import *
import torch

class TEDiscrete(ContextOfPastMeasureCalculatorDiscrete):

    def __init__(self, base = 2, destHistoryEmbedLength = 1, destEmbeddingDelay = 1, sourceHistoryEmbeddingLength = 1, sourceEmbeddingDelay = 1, delay = 1):
        ContextOfPastMeasureCalculatorDiscrete.__init__(self, base, destHistoryEmbedLength)
        self.base_power_l = power(base, sourceHistoryEmbeddingLength)

        self.init(base, destHistoryEmbedLength, destEmbeddingDelay, sourceHistoryEmbeddingLength, sourceEmbeddingDelay, delay)

    def init(self, base = 2, destHistoryEmbedLength = 1, destEmbeddingDelay = 1, sourceHistoryEmbeddingLength = 1, sourceEmbeddingDelay = 1, delay = 1):
        self.sourceNextPastCount = np.zeros(shape=(self.base_power_l, self.base, self.base_power_k), dtype=int)
        self.sourcePastCount = np.zeros(shape=(self.base_power_l, self.base_power_k), dtype=int)
        self.periodicBoundaryConditions = True
        self.base = 2
        self.startObservationTime = 1
        self.destEmbeddingDelay = destEmbeddingDelay
        if sourceHistoryEmbeddingLength <= 0:
            raise RuntimeError("Cannot have source embedding length of zero or less")
        self.sourceHistoryEmbedLength = sourceHistoryEmbeddingLength
        self.sourceEmbeddingDelay = sourceEmbeddingDelay
        self.delay = delay
        self.prev_te = 0.

        ##		// Check that we can convert the history value into an integer ok:
        ## if (sourceHistoryEmbedLength > Math.log(Integer.MAX_VALUE) / log_base) {
        ##	throw new RuntimeException("Base and source history combination too large");
        ##}

        self.maxShiftedSourceValue = np.zeros(shape=base, dtype=int)
        for v in range(0, base):
            self.maxShiftedSourceValue[v] = v * power(base, self.sourceHistoryEmbedLength - 1)

        # Create storage for extra counts of observations
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

        self.xs, self.xs_ = torch.empty(size=(0,), dtype=torch.uint8, device='cpu'), torch.empty(size=(0,),
                                                                                                 dtype=torch.uint8,
                                                                                                 device='cpu')  # array('h'), array('h')
        self.ys, self.ys_ = torch.empty(size=(0,), dtype=torch.uint8, device='cpu'), torch.empty(size=(0,),
                                                                                                 dtype=torch.uint8,
                                                                                                 device='cpu'),  # array('h'), array('h')

    def initialise(self):
        #ContextOfPastMeasureCalculatorDiscrete.initialise()
        #super(ContextOfPastMeasureCalculatorDiscrete, self).initialise()
        super(TEDiscrete, self).initialise()
        self.estimateComputed = False

        fill(self.sourceNextPastCount, 0)
        fill(self.sourcePastCount, 0)

        self.xs, self.xs_ = torch.empty(size=(0,),dtype=torch.uint8, device='cpu'), torch.empty(size=(0,),dtype=torch.uint8, device='cpu') #array('h'), array('h')
        self.ys, self.ys_ = torch.empty(size=(0,),dtype=torch.uint8, device='cpu'), torch.empty(size=(0,),dtype=torch.uint8, device='cpu'), #array('h'), array('h')


    def __len__(self):
        if(self.xs != None):
            return len(self.xs)
        return 0

    def add_source(self, src):
        self._add_item(True, src)

    def add_dest(self, dst):
        self._add_item(False, dst)

    def _add_item(self, src_or_dest, item: torch.Tensor):
        if item is not None and len(item) > 1:
            if src_or_dest:
                #self.xs = np.concatenate((self.xs, item))
                self.xs_ = item
            else:
                #self.ys = np.concatenate((self.ys, item))
                self.ys_ = item
        elif item is not None:
            if src_or_dest:
                self.xs_.append(item.item())
            else:
                self.ys_.append(item.item())

    # def addObservations(self, source: torch.Tensor, dest: torch.Tensor, startTime = 0, endTime = 0):
    #     if endTime == 0:
    #         endTime = len(dest) - 1
    #
    #     if ((endTime - startTime) - self.startObservationTime + 1 <= 0):
    #         # No observations to add
    #         return
    #
    #     if (endTime >= dest.shape[0]) or (endTime >= source.shape[0]):
    #         msg = "endTime {:d} must be <= length of input arrays (dest: {:d}, source: {:d})".format(endTime, dest.shape[0], source.shape[0])
    #         raise RuntimeError(msg)
    #
    #     self.observations += (endTime - startTime) - self.startObservationTime + 1
    #
    #     # Initialise and store the current previous values;
	# 	# one for each phase of the embedding delay.
	# 	# First for the destination:
    #     pastVal = np.zeros(shape=self.destEmbeddingDelay, dtype=int)
    #
    #     for d in range(0, self.destEmbeddingDelay):
    #         pastVal[d] = 0
    #
    #         for p in range(0, self.k-1):
    #             pastVal[d] += dest[startTime + self.startObservationTime + d -1 - (self.k - 1) * self.destEmbeddingDelay + p * self.destEmbeddingDelay]
    #             pastVal[d] *= self.base
    #
    #     sourcePastVal = np.zeros(shape=self.sourceEmbeddingDelay, dtype=int)
    #     for d in range(0, self.sourceEmbeddingDelay):
    #         sourcePastVal[d] = 0
    #         for p in range(0, self.sourceHistoryEmbedLength - 1):
    #             sourcePastVal[d] += source[startTime + self.startObservationTime + d - self.delay -
    #                                    (self.sourceHistoryEmbedLength - 1) * self.sourceEmbeddingDelay + p * self.sourceEmbeddingDelay]
    #             sourcePastVal[d] *= self.base
    #     #1. Count the tuples observed
    #     destVal = 0
    #     destEmbeddingPhase = 0
    #     sourceEmbeddingPhase = 0
    #     for r in range(startTime + self.startObservationTime, endTime + 1):
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
    #         #Now, update the combined embedding values and phases,
    #         # for this phase we back out the oldest value which we'll no longer need:
    #         if self.k > 0 :
    #             pastVal[destEmbeddingPhase] -= self.maxShiftedValue[dest[r-1-(self.k-1)*self.destEmbeddingDelay]]
    #             pastVal[destEmbeddingPhase] *= self.base
    #
    #         sourcePastVal[sourceEmbeddingPhase] -= self.maxShiftedSourceValue[source[r-self.delay-
    #                                                     (self.sourceHistoryEmbedLength-1)*self.sourceEmbeddingDelay]]
    #         sourcePastVal[sourceEmbeddingPhase] *= self.base
    #         # then update the phase
    #         destEmbeddingPhase = (destEmbeddingPhase + 1) % self.destEmbeddingDelay
    #         sourceEmbeddingPhase = (sourceEmbeddingPhase + 1) % self.sourceEmbeddingDelay

    def addOnlineObservationsLag1(self, source, dest, startTime = 0, endTime = 0):
        if endTime == 0:
            endTime = len(dest) - 1

        if ((endTime - startTime) - self.startObservationTime + 1 <= 0):
            # No observations to add
            return

        if (endTime >= len(dest) or endTime >= len(source)):
            msg = "endTime {:d} must be <= length of input arrays (dest: {:d}, source: {:d})".format(endTime,
                                                                                                     dest.shape[0],
                                                                                                     source.shape[0])
            raise RuntimeError(msg)

        self.observations += (endTime - startTime) - self.startObservationTime + 1

        # Initialise and store the current previous values;
        #  one for each phase of the embedding delay.
        # First for the destination:
        pastVal = np.zeros(shape=self.destEmbeddingDelay, dtype=int)
        pastVal[0] = 0

        sourcePastVal = np.zeros(shape=self.sourceEmbeddingDelay, dtype=int)

        sourcePastVal[0] = 0

        destVal = 0
        destEmbeddingPhase = 0
        sourceEmbeddingPhase = 0
        startIndex = startTime + self.startObservationTime
        endIndex = endTime + 1

        for r in list(range(startIndex, endIndex)):
            if self.k > 0:
                pastVal[destEmbeddingPhase] += dest[r - 1]
            sourcePastVal[sourceEmbeddingPhase] += source[r - self.delay]
            # Add to the count for this particular transition
            # (cell's assigned as above
            destVal = dest[r]
            thisPastVal = pastVal[destEmbeddingPhase]
            thisSourceVal = sourcePastVal[sourceEmbeddingPhase]
            self.update_counts(thisSourceVal, destVal, thisPastVal)
            self.update_past(source, dest, sourcePastVal, pastVal, destEmbeddingPhase, sourceEmbeddingPhase, r)
            # then update the phase
            destEmbeddingPhase = (destEmbeddingPhase + 1) % self.destEmbeddingDelay
            sourceEmbeddingPhase = (sourceEmbeddingPhase + 1) % self.sourceEmbeddingDelay

    def update_counts(self, thisSourceVal, destVal, thisPastVal):
        self.sourceNextPastCount[thisSourceVal][destVal][thisPastVal] += 1
        self.sourcePastCount[thisSourceVal][thisPastVal] += 1
        self.nextPastCount[destVal][thisPastVal] += 1
        self.pastCount[thisPastVal] += 1
        self.nextCount[destVal] += 1

    def update_past(self, source, dest, sourcePastVal, pastVal, destEmbeddingPhase, sourceEmbeddingPhase, r):
        # Now, update the combined embedding values and phases,
        # for this phase we back out the oldest value which we'll no longer need:
        if self.k > 0:
            pastVal[destEmbeddingPhase] -= self.maxShiftedValue[dest[r - 1 - (self.k - 1) * self.destEmbeddingDelay]]
            pastVal[destEmbeddingPhase] *= self.base

        sourcePastVal[sourceEmbeddingPhase] -= self.maxShiftedSourceValue[source[r - self.delay - ( self.sourceHistoryEmbedLength - 1) * self.sourceEmbeddingDelay]]
        sourcePastVal[sourceEmbeddingPhase] *= self.base

    def calcLocalTE(self):

        self.addOnlineObservationsLag1(self.xs_, self.ys_)
        temp_te = self.computeAverageLocalOfObservations()
        # in order not to keep the source timeseries in both locations
        # we are continuously appending here these, while in the netwowrk these are removed
        self.xs = torch.cat((self.xs, self.xs_.cpu()), dim=0)# np.concatenate((self.xs, self.xs_))
        self.ys = torch.cat((self.ys, self.ys_.cpu()), dim=0)# np.concatenate((self.ys, self.ys_))
        #self.prev_te = self.prev_te + temp_te
        #return self.prev_te / 2
        if temp_te < 1.0e-9:
            return 0.
        else:
            return temp_te
        #return np.float32(temp_te)

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
                        pinin1 = (float(pastVal) / float(self.observations))*(float(sourceVal) / float(self.observations))
                        pinjn = (float(pastVal) / float(self.observations)) * (float(sourceVal) / float(self.observations))
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
        self.std = math.sqrt(meanSqLocals - self.average * self.average)
        self.estimateComputed = True
        self.sum = te
        return te