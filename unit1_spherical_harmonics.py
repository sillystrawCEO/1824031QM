import numpy as np
import matplotlib.pyplot as plt
from math import factorial, pi, sqrt
from scipy.special import lpmv


def spherical_harmonic(l, m, theta, phi):
    normalisation = sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return normalisation * lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi)


def probability_density(l, m, theta, phi):
    return np.abs(spherical_harmonic(l, m, theta, phi)) ** 2


theta = np.linspace(0, np.pi, 120)
phi = np.linspace(0, 2 * np.pi, 120)
THETA, PHI = np.meshgrid(theta, phi)

quantum_numbers = [(1, 0), (1, 1)]

fig = plt.figure(figsize=(8, 4.5))

for index, (l, m) in enumerate(quantum_numbers):
    density = probability_density(l, m, THETA, PHI)
    radius = density / density.max()
    x = radius * np.sin(THETA) * np.cos(PHI)
    y = radius * np.sin(THETA) * np.sin(PHI)
    z = radius * np.cos(THETA)
    ax = fig.add_subplot(1, 2, index + 1, projection="3d")
    ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(radius), rstride=2, cstride=2, linewidth=0)
    ax.set_title(f"l = {l},  m = {m}")
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

plt.tight_layout()
plt.show()
