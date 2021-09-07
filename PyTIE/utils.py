import math

def power(base, power):
    result = 1
    absPower = int(math.fabs(float(power)))
    for p in range (0, absPower):
        result *= base

    if power < 0:
        # This will be zero for any base except 1 or -1
        result = 1 / result
    return result

def fill(matrix, value):
    rows = matrix.shape[0];
    for r in range(0, rows):
        cols = matrix[r].shape[0]
        for c in range(0, cols):
            matrix[r][c] = value


    # def isPowerOf2(int num):
	# 	bits = 0
	# 	shiftedValue = num
	#
	# 	for (int b = 0; b < Integer.SIZE; b ++) {
	# 		if ((shiftedValue & 0x01) > 0) {
	# 			// LSB is a 1
	# 			bits++;
	# 			if (bits > 1) {
	# 				// Encountered more than 1 bit set to 1.
	# 				// Is not a power of 2
	# 				return false;
	# 			}
	# 		}
	# 		// Shift a new bit down into the LSB
	# 		shiftedValue = shiftedValue >> 1;
	# 	}
	# 	// Post: num has either 1 bit set to 1 or 0 bits set to 1
	# 	if (bits == 0) {
	# 		return false;
	# 	}
	# 	return true;