import os
import random
import numpy as np
import cv2 
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
'''
Adapted from here: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
'''

class Noises(ImageOnlyTransform):
    def __init__(
            self, noise_type:str = 'speckle', SNR_dB:float = 5,
            always_apply=False,
            p=0.5,
    ):
        super(Noises, self).__init__(always_apply, p)
        self.noise_type = noise_type
        self.SNR_dB = SNR_dB

    def apply(self, image, **params):
        """
        Args:
            img (PIL Image): Image to apply color constancy on.

        Returns:
            Image: Image with color constancy.
        """
        img = self.noise(image, self.noise_type, self.SNR_dB)
        return img

    def noise(self, image, noise_type='speckle', SNR_dB=5):
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
            noisy = image + image * np.random.normal(mean,sigma,(row,col,ch))
            return noisy