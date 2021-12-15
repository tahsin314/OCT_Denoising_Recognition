import os
from tqdm.auto import tqdm as T
import cv2
import pandas as pd
from config import *
from speckle_filters import *
from scipy import misc


def noise(image, noise_type='speckle', SNR_dB=50):
    row,col,ch= image.shape
    mean = 0
    var = np.var(image) / (10 ** (SNR_dB / 10))
    sigma = var**0.5
    if noise_type == "gauss":
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_type == "s&p":
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_type == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_type =="speckle":
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

dirname = '../../data/OCT/oct2017/OCT2017 /train/NORMAL'
SEED = 42

images = [i for i in os.listdir(dirname) if '.DS_Store' not in i]
# print(images)
image = cv2.imread(f"{dirname}/{images[0]}", cv2.IMREAD_GRAYSCALE)
# image = misc.ascent()
# print(image.shape)
image = cv2.resize(image, (400, 400))
image_noise = noise(image.reshape(400, 400, 1), SNR_dB=35)
cv2.imwrite('img_noise.png', image_noise)
# image_filtered = lee_filter(image_noise.reshape(400, 400))
image_filtered = OSRAD(image_noise, 7, 0.8, 5, 1)
# image_filtered = SRAD(image, 5, 0.8, 5, 1)
mse = np.mean((image_filtered - image) ** 2)
psnr = 20*np.log10(255. / mse**0.5)
print(psnr)
print(image_filtered.max(), image_filtered.min())
cv2.imwrite('img_noise.png', image_noise)
cv2.imwrite('img_filtered.png', image_filtered)
# cv2.imwrite('img_filtered.png', 255.*image_filtered)
cv2.imwrite('img_original.png', image)