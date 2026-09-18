from dataclasses import dataclass
import numpy as np

@dataclass
class FDM_HeatModel:
    """struct con todos los datos del modelo (constantes, esquema numérico, etc.)"""
    
    nnodes: int = 10
    """Cantidad total de nodos de la malla"""

    k: np.ndarray | None = None
    """Conductividad térmica del material, k(x,y)"""

    c: np.ndarray | None = None
    """Constante de reacción del sistema, c(x,y)"""

    G: np.ndarray | None = None
    """Fuente de calor volumétrica, G(x,y)"""

    ts: int = -1
    """
    Selección del esquema temporal:
        0 = Explícito
        1 = Implícito
        otro valor = Estado estacionario
    """

    rho: float = 1.0
    """Densidad del material"""

    cp: float = 1.0
    """Calor específico a presión constante"""

    maxit: int = 1000
    """Cantidad máxima de iteraciones para esquemas temporales"""
    
    tol: float = 1e-6
    """Tolerancia de error relativo entre iteraciones"""

    dt: float = 0.01
    """Paso temporal del esquema implícito"""

    PHI_n: np.ndarray | None = None
    """Condición inicial para esquemas temporales. Vector con un valor por cada nodo"""
    
    def __post_init__(self):
        if self.k is None:
            self.k = np.ones(self.nnodes)
        elif len(self.k) != self.nnodes:
            raise ValueError("k debe tener nnodes elementos")

        if self.c is None:
            self.c = np.zeros(self.nnodes)
        elif len(self.c) != self.nnodes:
            raise ValueError("c debe tener nnodes elementos")

        if self.G is None:
            self.G = np.zeros(self.nnodes)
        elif len(self.G) != self.nnodes:
            raise ValueError("G debe tener nnodes elementos")

        if self.PHI_n is None:
            self.PHI_n = np.zeros(self.nnodes)
        elif len(self.PHI_n) != self.nnodes:
            raise ValueError("PHI_n debe tener nnodes elementos")