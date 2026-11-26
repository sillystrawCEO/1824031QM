import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0
mass = 1.0
length = 1.0
points = 1000

x = np.linspace(0, length, points + 2)
h = x[1] - x[0]

diagonal = np.full(points, hbar ** 2 / (mass * h ** 2))
off = np.full(points - 1, -hbar ** 2 / (2 * mass * h ** 2))
hamiltonian = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)

energies, states = np.linalg.eigh(hamiltonian)

print("Particle in a box")
for n in range(1, 6):
    analytical = n ** 2 * np.pi ** 2 * hbar ** 2 / (2 * mass * length ** 2)
    print(f"n = {n}:  numerical = {energies[n - 1]:.4f}   analytical = {analytical:.4f}")

plt.figure(figsize=(8, 5))
for n in range(3):
    wave = states[:, n] / np.sqrt(h)
    if wave.sum() < 0:
        wave = -wave
    full_wave = np.concatenate(([0.0], wave, [0.0]))
    plt.plot(x, full_wave, label=f"n = {n + 1}")
plt.title("Particle in a box: first three wavefunctions")
plt.xlabel("x")
plt.ylabel("psi(x)")
plt.legend()
plt.grid(True)
plt.show()
