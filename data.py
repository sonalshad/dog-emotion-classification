import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class DogsDataset(Dataset):
    def __init__(self, path, label_csv, transforms=False, valid_extensions=('.jpg', '.jpeg', '.png')):
        self.path_to_images = Path(path)
        self.transforms = transforms

        df = pd.read_csv(label_csv, header = None)
        self.labels = dict(zip(df[0], df[1]))

        # Create a list of file paths, filtering by valid extensions and CSV filenames
        self.files = [
            self.path_to_images / filename for filename in self.labels.keys()
            if (self.path_to_images / filename).exists() and Path(filename).suffix.lower() in valid_extensions
        ]

        # Create an array of labels
        self.y = np.array([self.labels[file.name] for file in self.files], dtype=int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        path = str(self.files[idx])
        x = cv2.imread(path).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255

        if self.transforms:
            rdeg = (np.random.random() - 0.5) * 20
            x = rotate_cv(x, rdeg)
            x = random_crop(x)
            if np.random.random() > 0.5:
                x = np.fliplr(x).copy()
            x = cv2.resize(x, (224, 224))

        else:
            x = center_crop(x)
            x = cv2.resize(x, (224, 224))

        vid = path.split("/")[-1].split("-")[0]

        # x = normalize(x)
        return np.rollaxis(x, 2), self.y[idx], vid

# Define transformations for augmentation
def crop(im, r, c, target_r, target_c): return im[r:r+target_r, c:c+target_c]

def random_crop(x, r_pix=8):
    """Returns a random crop that maintains minimum size of 224x224."""
    r, c, *_ = x.shape
    if r > 224 + 2 * r_pix and c > 224 + 2 * r_pix:
        c_pix = round(r_pix * c / r)
        rand_r = random.uniform(0, 1)
        rand_c = random.uniform(0, 1)
        start_r = np.floor(2 * rand_r * r_pix).astype(int)
        start_c = np.floor(2 * rand_c * c_pix).astype(int)
        return crop(x, start_r, start_c, r - 2 * r_pix, c - 2 * c_pix)
    else:
        return x  # If smaller, return original

def center_crop(x, r_pix=8):
    r, c, *_ = x.shape
    if r > 224 + 2 * r_pix and c > 224 + 2 * r_pix:
        c_pix = round(r_pix * c / r)
        return crop(x, r_pix, c_pix, r - 2 * r_pix, c - 2 * c_pix)
    else:
        return x  # If smaller, return original

def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode,
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def normalize(im):
    """Normalizes images with ImageNet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]