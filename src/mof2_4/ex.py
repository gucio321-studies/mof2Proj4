from scipy import constants as const
import numpy as np
from matplotlib import pyplot as plt

def ex2():
    b = base()
    V = b.V(b.x)
    b.solve()
    psiSq = np.abs(b.psi)**2

    fig, ax = plt.subplots()
    
    ax.plot(b.x*b.A0, psiSq)
    ax.set_xlabel("położenie [m]")
    ax.set_ylabel("$|\\Psi|^2$")
    ax.grid()

    ax = ax.twinx()
    ax.plot(b.x*b.A0, base.au_to_meV(V), color='red')
    ax.set_ylabel("Energia [meV]")

    fig.savefig("ex2.svg", format="svg")
    plt.close(fig)
    

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
        v = self.meV_to_au(-3.0)
        return np.where((self.l <= x) & (x <= 2 * self.l), v, 0.)

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
        eiqx1 = np.exp(1j*self.q*x1)
        eiqx2 = np.exp(1j*self.q*x2)

        A = (psi1*eiqx1 - psi2*eiqx2) /(eiqx1**2-eiqx2**2)
        B = -(-psi2*eiqx1+psi1*eiqx2)*eiqx1*eiqx2/(eiqx1**2-eiqx2**2)

        return (A, B)
