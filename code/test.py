import pandas as pd
import tensorflow as tf
import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.version.cuda)         # Displays the CUDA version used
print('Tensorflow check')
print(tf.test.is_gpu_available())
import torch.c