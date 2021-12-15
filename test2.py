from functools import partial
import os
from config import *
from speckle_filters import OSRAD, SRAD, lee_filter
from utils import *
from p_tqdm import p_map as PT
from tqdm import tqdm as T
from functools import partial

data_path = '../../data/OCT/oct2017/OCT2017 /'
noise_path = '../../data/OCT/oct2017/OCT2017_noise/'
lee_path = '../../data/OCT/oct2017/OCT2017_lee/'
srad_path = '../../data/OCT/oct2017/OCT2017_srad/'
osrad_path = '../../data/OCT/oct2017/OCT2017_osrad/'
os.makedirs(noise_path, exist_ok=True)
test_df = test_df[:2000]
# for i in T([25, 30, 35, 40, 45]):
def noisy_image(i):
    os.makedirs(f"{noise_path}/{i}/", exist_ok=True)
    test_df['id_noise'] = test_df['id'].map(lambda x: x.replace(data_path, f"{noise_path}/{i}/"))
    for img_id, noise_id in T(test_df[['id', 'id_noise']].values, total=len(test_df)):
        new_path = '/'.join(i for i in noise_id.split('/')[:-1])
        os.makedirs(new_path, exist_ok=True)
        img = cv2.imread(img_id)
        img_noise = noise(img, SNR_dB=i)
        cv2.imwrite(noise_id, img_noise)
    # image = cv2.imread(test_df.id.values[0])

# PT(noisy_image, [35, 40, 45, 50])

def filter_image(i, filter_path=osrad_path, filter_func=OSRAD):
    os.makedirs(f"{filter_path}/{i}/", exist_ok=True)
    test_df['id_noise'] = test_df['id'].map(lambda x: x.replace(data_path, f"{noise_path}/{i}/"))
    test_df['id_filter'] = test_df['id'].map(lambda x: x.replace(data_path, f"{'id_filter'}/{i}/"))
    for img_id, noise_id, filter_id in T(test_df[['id', 'id_noise', 'id_filter']].values, total=len(test_df)):
        new_path = '/'.join(i for i in filter_id.split('/')[:-1])
        os.makedirs(new_path, exist_ok=True)
        img = cv2.imread(noise_id, cv2.IMREAD_GRAYSCALE)
        img_filtered = filter_func(img)
        cv2.imwrite(filter_id, img_filtered)

# partial(filter_image, filter_path=lee_path)
# partial(filter_image, filter_func=lee_filter)
# PT(filter_image, [35, 40, 45, 50])
os.makedirs('noise_folder', exist_ok=True)

def noisy_image_gen(img_id, i):
    img = cv2.imread(img_id)
    img_noise = noise(img, SNR_dB=i)
    return img_noise
    # cv2.imwrite(f"noise_folder/{img_id.split('.')[0]}_{i}.png", img_noise)
img_stack = None
for i in T([30, 36, 39, 38.2, 40]):
    noisy_img = noisy_image_gen('DME-30521-18.jpeg', i)
    img_stack = noisy_img if img_stack is None else np.hstack((img_stack, noisy_img))

cv2.imwrite('noise_folder/noisy_img_4.png', img_stack)