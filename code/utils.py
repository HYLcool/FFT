import matplotlib.pyplot as plt
import math

# show multi-images in a window
def showImgN(imgs, titles = None):
	# grids definition
	N = int(len(imgs))
	h = int(math.sqrt(N) + 0.5)
	w = h + (1 if h * h < N else 0)
	for i in range(N):
		img = imgs[i]
		title = titles[i] if titles != None else 'img ' + str(i + 1)
		# show images
		plt.subplot(h, w, i + 1)
		plt.imshow(img)
		plt.title(title)
		plt.xticks([])
		plt.yticks([])
	plt.show()

# Is n equal to 2^(integer)
def exp2int(n):
	return (n & (n - 1)) == 0
# get 2^(integer) that is >= n
def geExp2(n):
	i = math.log2(n)
	i = math.ceil(i)
	return int(pow(2, i))