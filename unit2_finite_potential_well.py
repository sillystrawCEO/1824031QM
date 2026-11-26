import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0
mass = 1.0
width = 2.0
depth = 10.0
x_max = 8.0
points = 2000

x = np.linspace(-x_max, x_max, points + 2)
h = x[1] - x[0]
inner = x[1:-1]
potential = np.where(np.abs(inner) <= width / 2, -depth, 0.0)

diagonal = hbar ** 2 / (mass * h ** 2) + potential
off = np.full(points - 1, -hbar ** 2 / (2 * mass * h ** 2))
hamiltonian = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)

energies, states = np.linalg.eigh(hamiltonian)
count = np.sum(energies < 0)

print(f"Finite potential well: width = {width}, depth = {depth}")
for n in range(count):
    print(f"State {n + 1}:  E = {energies[n]:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(inner, potential, "k", linewidth=2, label="potential")
for n in range(count):
    wave = states[:, n] / np.sqrt(h)
    if wave.sum() < 0:
        wave = -wave
    plt.plot(inner, wave + energies[n], label=f"n = {n + 1}")
    plt.axhline(energies[n], color="gray", linestyle="--", linewidth=0.8)
plt.title(f"Finite potential well: width = {width}, depth = {depth}")
plt.xlabel("x")
plt.ylabel("Energy / wavefunction")
plt.legend()
plt.grid(True)
plt.show()
