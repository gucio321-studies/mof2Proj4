from scipy import constants as const
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def ex2(E=7, out="ex2"):
    b = base(E=E)
    V = b.V(b.x)
    b.solve()
    psiSq = np.abs(b.psi) ** 2

    A, B = b.getProb()
    print(f"{out.title()}: Otrzymano wartości A={A} i B={B}")
    x_lt = b.x[: b.N - 1]
    psi_lt_A = A * np.exp(1j * b.q * x_lt)
    psi_lt_A_sq = np.abs(psi_lt_A) ** 2
    psi_lt_B = B * np.exp(-1j * b.q * x_lt)
    psi_lt_B_sq = np.abs(psi_lt_B) ** 2

    psi_lt = psi_lt_A + psi_lt_B
    psi_lt_sq = np.abs(psi_lt) ** 2

    fig, ax = plt.subplots()

    ax.plot(b.x * b.A0, psiSq, label="$|\\Psi|^2$ numeryczne")
    ax.plot(x_lt * b.A0, psi_lt_sq, label="$|\\Psi_<|^2$")
    ax.plot(x_lt * b.A0, psi_lt_A_sq, label="$|\\Psi_{in}|^2$")
    ax.plot(x_lt * b.A0, psi_lt_B_sq, label="$|\\Psi_{R}|^2$", color="purple")
    ax.set_xlabel("położenie [m]")
    ax.set_ylabel("$|\\Psi|^2$")
    ax.grid()
    ax.legend()

    ax = ax.twinx()
    ax.plot(b.x * b.A0, base.au_to_meV(V), color="red")
    ax.set_ylabel("Energia [meV]")

    fig.savefig(f"{out}.svg", format="svg")
    plt.close(fig)


def ex3():
    # Polecenie nakazuje narysować (0, 60], ale moim zdaniem logspace jest tutaj ciekawszy
    grid = 200
    Es = np.logspace(-2, 3, grid)
    Rs = np.zeros(grid)
    for i in range(grid):
        b = base(E=Es[i])
        b.solve()
        A, B = b.getProb()
        Rs[i] = np.abs(B / A) ** 2

    fig, ax = plt.subplots()
    ax.plot(Es, Rs, label="R(E)")
    ax.plot(Es, 1 - Rs, label="T(E)")
    ax.set_xscale("log")
    ax.set_xlabel("E [meV]")
    ax.legend()
    ax.grid()
    fig.savefig("ex3.svg", format="svg")
    plt.close(fig)

    return (Es, Rs)


def ex4(Es, Rs):
    count = 3
    minimasI, _ = find_peaks(-Rs)
    minimasI = minimasI[:count] # lets say we want only 3 energies
    Es = Es[minimasI]
    for i in range(count):
        print(f"Ex4: Resonance for E={Es[i]}, T={Rs[i]}")
        ex2(E=Es[i], out=f"ex4_resonance_{i}")



class base:
    def __init__(self, dx=0.5e-9, N=100, m=0.067 * const.m_e, E=7):
        self.A0 = const.physical_constants["Bohr radius"][0]
        self.N = N
        self.dx = dx / self.A0
        self.l = self.dx * (self.N - 1)
        self.m = m / const.m_e
        self.E = self.meV_to_au(E)
        self.q = np.sqrt(2 * self.m * self.E)
        self.__init_grid()
        self.solve()

    def __init_grid(self):
        self.x = np.linspace(0, 3 * self.l, 3 * self.N - 2)

    def V(self, x):
        v = self.meV_to_au(-30.0)
        return np.where((self.l <= x) & (x <= 2 * self.l), v, 0.0)

    @staticmethod
    def meV_to_au(meV):
        hartree_eV = const.physical_constants["Hartree energy in eV"][0]
        return meV * 1e-3 / hartree_eV

    @staticmethod
    def au_to_meV(au):
        hartree_eV = const.physical_constants["Hartree energy in eV"][0]
        return au * hartree_eV * 1e3

    def solve(self):
        self.psi = np.zeros(3 * self.N - 2, dtype=complex)
        self.psi[3 * self.N - 3] = 1
        self.psi[3 * self.N - 4] = np.exp(-1j * self.q * self.dx)
        for i in range(3 * self.N - 5, -1, -1):
            x = self.x[i + 1]
            self.psi[i] = (
                -2 * self.m * (self.E - self.V(x)) * self.dx**2 * self.psi[i + 1]
                - self.psi[i + 2]
                + 2 * self.psi[i + 1]
            )

    def getProb(self):
        """
        Returtns A and B coefficients
        This assumes solve() was already called
        """
        x1, x2 = self.x[0], self.x[1]
        psi1, psi2 = self.psi[0], self.psi[1]
        eiqx1 = np.exp(1j * self.q * x1)
        eiqx2 = np.exp(1j * self.q * x2)

        A = (psi1 * eiqx1 - psi2 * eiqx2) / (eiqx1**2 - eiqx2**2)
        B = -(-psi2 * eiqx1 + psi1 * eiqx2) * eiqx1 * eiqx2 / (eiqx1**2 - eiqx2**2)

        return (A, B)
