import matplotlib.pyplot as plt
import numpy as np
from utils import *
import math
import cmath

# W(b)^i = exp(-j*2*pi*(i)/b)
def W(index, basic):
	return cmath.exp(-1j * 2 * cmath.pi * (index) / basic)
# padding f to geExp2
def padding(f):
	if len(f.shape) > 2:
		cW, cH, c = f.shape
	else:
		cW, cH = f.shape
		c = 1
	rW = geExp2(cW)
	rH = geExp2(cH)
	if rW != cW:
		f = np.concatenate((f, np.zeros((rW, cH, c)) if c > 1 else np.zeros((rW, rH))))
	if rH != cH:
		f = np.concatenate((f, np.zeros((rW, rH, c)) if c > 1 else np.zeros((rW, rH))), axis = 1)
	return f

# 1-D FFT
def fft1d(f):
	N = int(f.shape[0])
	if N == 1:
		return f
	else:
		Feven = fft1d(f[range(0, N, 2)])
		Fodd = fft1d(f[range(1, N, 2)])
		res = np.zeros(f.shape, dtype = complex)
		K = int(N / 2)
		for u in range(K):
			res[u] = Feven[u] + Fodd[u] * W(u, N)
			res[u + K] = Feven[u] - Fodd[u] * W(u, N)
		return res

'''
	FFT: Fast Fourier Transform
	Input: an image f
	Output: transformed F
'''
def FFT(f):
	# get N, M and c
	if len(f.shape) > 2:
		M, N, c = f.shape
	else:
		M, N = f.shape
		c = 1
	# padding to 2^integer
	f = padding(f)
	if len(f.shape) > 2:
		newM, newN, c = f.shape
	else:
		newM, newN = f.shape
		c = 1
	# result
	res = np.zeros(f.shape, dtype = complex)

	for i in range(c):
		if c == 1:
			fc = f
		else:
			fc = f[:,:,i]
		rc = np.zeros(fc.shape, dtype = complex)

		# 1-D FFT twice
		for u in range(newM):
			rc[u, :] = fft1d(fc[u, :])
		for v in range(newN):
			rc[:, v] = fft1d(rc[:, v])

		if c == 1:
			res = rc
		else:
			res[:,:,i] = rc
	return res
'''
	FFTShift: shift (0, 0) to center
'''
def FFTShift(f):
	if len(f.shape) > 2:
		M, N, c = f.shape
	else:
		M, N = f.shape
		c = 1
	res = np.zeros(f.shape, dtype = complex)

	for i in range(c):
		if c == 1:
			fc = f
		else:
			fc = f[:,:,i]
		rc = np.zeros(fc.shape, dtype = complex)

		tmp = np.zeros(fc.shape, dtype = complex)
		# move horizontally
		for u in range(M):
			for v in range(N):
				newV = int(v + N / 2) % N
				tmp[u, newV] = fc[u, v]
		# move vertically
		for v in range(N):
			for u in range(M):
				newU = int(u + M / 2) % M
				rc[newU, v] = tmp[u, v]
		if c == 1:
			res = rc
		else:
			res[:, :, i] = rc
	return res

'''
	iFFT: inverse FFT
	Notice: exp(-j) = exp(j).conjugate()
'''
def iFFT(F):
	# get N, M and c
	if len(F.shape) > 2:
		M, N, c = F.shape
	else:
		M, N = F.shape
		c = 1
	# result
	res = np.zeros(F.shape, dtype = complex)

	for i in range(c):
		if c == 1:
			fc = F
		else:
			fc = F[:,:,i]
		rc = np.zeros(fc.shape, dtype = complex)

		# 1-D FFT twice
		for u in range(M):
			rc[u, :] = fft1d(fc[u, :].conj())
		for v in range(N):
			rc[:, v] = fft1d(rc[:, v].conj())

		if c == 1:
			res = rc
		else:
			res[:,:,i] = rc
	res = res.conj() / (M * N)
	return res

def main():
	f = plt.imread('../img/fft.tif')
	print(f.shape)
	# my implementation
	F = FFT(f)
	FS = FFTShift(F)
	iF = iFFT(FS)
	F = np.log(np.abs(F) + np.ones(F.shape))
	FS = np.log(np.abs(FS) + np.ones(FS.shape))
	iF = np.abs(iF)
	# numpy implementation
	F1 = np.fft.fft2(f)
	FS1 = np.fft.fftshift(F1)
	iF1 = np.fft.ifft2(FS1)
	F1 = np.log(np.abs(F1) + np.ones(F.shape))
	FS1 = np.log(np.abs(FS1) + np.ones(FS.shape))
	iF1 = np.abs(iF1)
	showImgN([f, F, FS, iF, F1, FS1, iF1], ('original', 'FFT', 'FFT Shift', 'inverse FFT', 'FFT np', 'FFT Shift np', 'inverse FFT np'))

if __name__ == '__main__':
	main()