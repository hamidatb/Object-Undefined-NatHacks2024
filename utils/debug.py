import matplotlib.pyplot as plt
import numpy as np
from train_quadrant_model import load_samples

samples, labels = load_samples()
samples = np.array(samples)
labels = np.array(labels)

plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap='viridis', s=10)
plt.colorbar(label='Quadrant Label')
plt.xlabel('Left Eye X')
plt.ylabel('Left Eye Y')
plt.title('Pupil Position Distribution')
plt.show()

