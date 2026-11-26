import numpy as np
import matplotlib.pyplot as plt

hc = 197.3
mc2 = 940.0
hbar2_over_2m = hc ** 2 / (2 * mc2)
k = 100.0
r_max = 10.0
points = 1000

r = np.linspace(0, r_max, points + 2)
h = r[1] - r[0]
inner = r[1:-1]
off = np.full(points - 1, -hbar2_over_2m / h ** 2)


def ground_state(b):
    potential = 0.5 * k * inner ** 2 + (1 / 3) * b * inner ** 3
    diagonal = 2 * hbar2_over_2m / h ** 2 + potential
    hamiltonian = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    energies, states = np.linalg.eigh(hamiltonian)
    wave = states[:, 0] / np.sqrt(h)
    if wave.sum() < 0:
        wave = -wave
    return energies[0], wave


print("Anharmonic oscillator (s-wave, particle mass = 940 MeV/c^2)")
plt.figure(figsize=(8, 5))
for b in [0, 10, 30]:
    energy, wave = ground_state(b)
    print(f"b = {b} MeV/fm^3:  ground state energy = {energy:.2f} MeV")
    plt.plot(inner, wave, label=f"b = {b}")
plt.title("Anharmonic oscillator: ground state wavefunctions")
plt.xlabel("r (fm)")
plt.ylabel("u(r)")
plt.legend()
plt.grid(True)
plt.show()
