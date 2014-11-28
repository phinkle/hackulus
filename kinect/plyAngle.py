
def squaredDistance(pts3Da, pts3Db):
	if len(pts3Da) != len(pts3Db):
		raise Error()

	dSq = 0.0
	for index in xrange(len(pts3Da)):
		# Wow this is horrendously beautiful
		dSq += squaredDistance(pts3Da[index], pts3Db[index])

	return dSq

def rotation(theta):
	# consider negating the sins.. Whichever works
	return np.matrix(
		[
		[cos(theta), 0, -sin(theta)],
		[0, 0, 0],
		[sin(theta), 0, cos(theta)]
		]);

def transform(matrix, pts):
	newPts = []

	for point in pts:
		newPts.append(matrix * point)

	return newPts

def near(a, b, error):
	return abs(a - b) <= error

def findAngle(pts3Da, pts3Db, centerOfRot):
	if len(centerOfRot) != 3:
		raise Error()

	if len(pts3Da) == 0 or len(pts3Da) != len(pts3Db):
		raise Error()

	if len(pts3Da[0]) != 3:
		rasie Error()

	# degree - transfer to radians if sin/cos take radians
	minAngleOff = 45

	lo = -minAngleOff
	hi = minAngleOff

	# change the error however you would like.
	# the smaller the error, the more accurate and longer it takes
	while !near(lo, hi, 1.0 / pow(10, 6)):
		third = (hi - lo) / 3.0
		x = lo + third
		y = lo + third * 2

		left = squaredDistance(pts3Da, transform(rotation(x), pts3Db))
		right = squaredDistance(pts3Da, transform(rotation(y), pts3Db))

		if left > right:
			lo = x
		else:
			hi = y

	return lo

def combine(plys, centerOfRot):
	if len(plys) == 0:
		raise Exception()

	newPlys = [plys[0]]

	for index in xrange(len(plys) - 1):
		imageA = removeDepth(plys[0])
		imageB = removeDepth(plys[1])

		# Uses homography project to find matching points
		# produces 2d tuples
		ptsA, ptsB = featureMatch(imageA, imageB)

		# Puts the depth back into the images from the original
		# produces 3d tuples
		addDepth(imageA, ptsA)
		addDepth(imageB, ptsB)

		# Puts images back to origin for rotations
		translate(-centerOfRot, ptsA)

		rotAngle = findAngle(ptsA, ptsB, centerOfRot)
		newPly = transform(rotation(rotAngle), ptsB)
		newPlys.append(newPly)

	return newPlys
