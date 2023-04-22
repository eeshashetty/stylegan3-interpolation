# Imports
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os

# Set Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
import sys
sys.path.append("stylegan3")

path = "stylegan3-t-ffhq-1024x1024.pkl"
with open(path, 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)

## Helper Functions

# Generate images given a latent code
def generate_image_from_z(z):
  c = None # class labels (not used in this example)
  images = G(z, c) # NCHW, float32, dynamic range [-1, +1], no truncation
  images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  return images

# Generate images given a random seed
def generate_image_random(seed):
  z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
  
  label = torch.zeros([1, G.c_dim], device=device)

  img = G(z, label)
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  
  return img, z

# Interpolate between latent codes
def linear_interpolate(z1, z2, alpha):
  return z1 * alpha + z2 * (1 - alpha)

# Generate Interpolated Image
def generate_interpolation(z1, z2, alpha):
  z_interpolated = linear_interpolate(z1, z2, alpha) 
  images = generate_image_from_z(z_interpolated)
  return images[0].cpu().numpy()