import numpy as np

def generar_DIR(frontera, xnode, funcion):
    """
    Genera la matriz DIR correspondiente a una frontera.

    Parámetros
    ----------
    frontera : array_like
        Índices de los nodos de la frontera.

    xnode : ndarray, shape (nnodes, 2)
        Coordenadas de todos los nodos de la malla.

    funcion : callable
        Función que recibe (x, y) y devuelve el valor
        de la condición de Dirichlet.

    Retorna
    -------
    DIR : ndarray, shape (n, 2)
        Columna 1: índice del nodo.
        Columna 2: valor de la condición.
    """

    frontera = np.asarray(
        frontera,
        dtype=int
    )

    puntos = xnode[frontera]

    x = puntos[:, 0]
    y = puntos[:, 1]

    valores = np.asarray(
        funcion(x, y),
        dtype=float
    )

    if valores.ndim == 0:
        valores = np.full(
            len(frontera),
            valores,
            dtype=float
        )

    if len(valores) != len(frontera):
        raise ValueError(
            "La función de Dirichlet debe devolver un valor "
            "por cada nodo de la frontera."
        )

    DIR = np.column_stack([
        frontera,
        valores
    ])

    return DIR


def generar_NEU(frontera, xnode, funcion, normal):
    """
    Genera la matriz NEU correspondiente a una frontera.

    Parámetros
    ----------
    frontera : array_like
        Índices de los nodos de la frontera.

    xnode : ndarray, shape (nnodes, 2)
        Coordenadas de todos los nodos de la malla.

    funcion : callable
        Función que recibe (x, y) y devuelve el valor del
        flujo térmico q.

    normal : int
        Dirección de la normal exterior:

            1 = S = ( 0, -1)
            2 = E = ( 1,  0)
            3 = N = ( 0,  1)
            4 = W = (-1,  0)

    Retorna
    -------
    NEU : ndarray, shape (n, 3)
        Columna 1: índice del nodo.
        Columna 2: flujo q.
        Columna 3: dirección de la normal.
    """

    frontera = np.asarray(
        frontera,
        dtype=int
    )

    if normal not in (1, 2, 3, 4):
        raise ValueError(
            "La normal debe ser 1 (S), 2 (E), "
            "3 (N) o 4 (W)."
        )

    puntos = xnode[frontera]

    x = puntos[:, 0]
    y = puntos[:, 1]

    valores = np.asarray(
        funcion(x, y),
        dtype=float
    )

    if valores.ndim == 0:
        valores = np.full(
            len(frontera),
            valores,
            dtype=float
        )

    if len(valores) != len(frontera):
        raise ValueError(
            "La función de Neumann debe devolver un valor "
            "por cada nodo de la frontera."
        )

    direcciones = np.full(
        len(frontera),
        normal,
        dtype=int
    )

    NEU = np.column_stack([
        frontera,
        valores,
        direcciones
    ])

    return NEU


def generar_ROB(frontera, xnode, funcion_h, funcion_phi_inf, normal):
    """
    Genera la matriz ROB correspondiente a una frontera.

    Parámetros
    ----------
    frontera : array_like
        Índices de los nodos de la frontera.

    xnode : ndarray, shape (nnodes, 2)
        Coordenadas de todos los nodos de la malla.

    funcion_h : callable
        Función que recibe (x, y) y devuelve h.

    funcion_phi_inf : callable
        Función que recibe (x, y) y devuelve phi_inf.

    normal : int
        Dirección de la normal exterior:

            1 = S = ( 0, -1)
            2 = E = ( 1,  0)
            3 = N = ( 0,  1)
            4 = W = (-1,  0)

    Retorna
    -------
    ROB : ndarray, shape (n, 4)
        Columna 1: índice del nodo.
        Columna 2: h.
        Columna 3: phi_inf.
        Columna 4: dirección de la normal.
    """

    frontera = np.asarray(
        frontera,
        dtype=int
    )

    if normal not in (1, 2, 3, 4):
        raise ValueError(
            "La normal debe ser 1 (S), 2 (E), "
            "3 (N) o 4 (W)."
        )

    puntos = xnode[frontera]

    x = puntos[:, 0]
    y = puntos[:, 1]

    h = np.asarray(
        funcion_h(x, y),
        dtype=float
    )

    phi_inf = np.asarray(
        funcion_phi_inf(x, y),
        dtype=float
    )

    if h.ndim == 0:
        h = np.full(
            len(frontera),
            h,
            dtype=float
        )

    if phi_inf.ndim == 0:
        phi_inf = np.full(
            len(frontera),
            phi_inf,
            dtype=float
        )

    if len(h) != len(frontera):
        raise ValueError(
            "La función h debe devolver un valor "
            "por cada nodo de la frontera."
        )

    if len(phi_inf) != len(frontera):
        raise ValueError(
            "La función phi_inf debe devolver un valor "
            "por cada nodo de la frontera."
        )

    direcciones = np.full(
        len(frontera),
        normal,
        dtype=int
    )

    ROB = np.column_stack([
        frontera,
        h,
        phi_inf,
        direcciones
    ])

    return ROB