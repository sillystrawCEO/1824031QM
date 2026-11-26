import numpy as np
import matplotlib.pyplot as plt

hc = 1973.0
mc2 = 0.511e6
e_squared = 3.795 ** 2
hbar2_over_2m = hc ** 2 / (2 * mc2)
rydberg = e_squared ** 2 / (4 * hbar2_over_2m)

r_max = 40.0
points = 2000

r = np.linspace(0, r_max, points + 2)
h = r[1] - r[0]
inner = r[1:-1]
potential = -e_squared / inner

diagonal = 2 * hbar2_over_2m / h ** 2 + potential
off = np.full(points - 1, -hbar2_over_2m / h ** 2)
hamiltonian = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)

energies, states = np.linalg.eigh(hamiltonian)

print("Hydrogen atom, s-wave")
for n in range(1, 5):
    analytical = -rydberg / n ** 2
    print(f"n = {n}:  numerical = {energies[n - 1]:.4f} eV   analytical = {analytical:.4f} eV")

plt.figure(figsize=(8, 5))
for n in range(2):
    wave = states[:, n] / np.sqrt(h)
    if wave.sum() < 0:
        wave = -wave
    plt.plot(inner, wave, label=f"n = {n + 1}")
plt.title("Hydrogen atom: radial function u(r) for s-wave states")
plt.xlabel("r (Angstrom)")
plt.ylabel("u(r)")
plt.legend()
plt.grid(True)
plt.show()
