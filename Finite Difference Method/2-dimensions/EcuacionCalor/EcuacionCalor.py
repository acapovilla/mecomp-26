import numpy as np
from .model import FDM_HeatModel

def fdm2d_initialize(nnodes: int):
    """
    **Descripción**: módulo para inicializar las variables principales del
    sistema, las cuales se utilizarán para almacenar los datos calculados y 
    ensamblados por el método numérico. (CAMBIAR: Se inicializan como sparse 
    para optimizar el rendimiento general del método numérico.)
    
    **Entrada**:
    
    `nnodes`: cantidad de nodos de la malla.
    
    **Salida**:
    
    `K`: matriz del sistema (difusión + reacción)
    
    `F`: vector de flujo térmico.
    """
    
    K = np.zeros((nnodes, nnodes))
    F = np.zeros((nnodes,))
    
    return K, F

def fdm2d_neighbors(icone: np.ndarray):
    """
    **Descripción**: módulo para armar la matriz de vecindad (`neighb`).
    La malla del dominio se encuentra formada por elementos rectangulares.
    Cada uno de los 4 nodos que forman el elemento se enumeran en sentido
    antihorario empezando con la esquina inferior izquierda. A partir de esta
    conectividad (almacenada en icone) se arma la matriz de vecindad indicando
    para cada nodo sus nodos vecinos. Si un nodo no tiene algún vecino, el 
    índice correspondiente se guarda como -1. Existen 3 tipos de nodos:
    - Nodos esquina: los cuales se conectan con otros 2 nodos.
    - Nodos de borde: los cuales se conectan con otros 3 nodos.
    - Nodos interiores: los cuales se conectan con otros 4 nodos.
    
    Para un nodo cualquiera [P] se define la siguiente vecindad:
    ```
    .                   [N] (North - Norte)
    .                    |
    .(West - Oeste) [W]-[P]-[E] (East - Este)
    .                    |
    .                   [S] (South - Sur)
    ```
    
    **Entrada**:
    
    `icone`: matriz de conectividad. Cada fila de la matriz indica la
    conectividad de un elemento rectangular, comenzando por el extremo inferior
    izquierdo y recorriendo el elemento en sentido antihorario.
    
    **Salida**:
    
    `neighb`: matriz de vecindad.
    """
    # El tamaño se infiere a partir del máximo índice encontrado en 'icone'
    _max_idx = np.max(icone)
    
    # El tamaño sería (cantidad de puntos, 4) -> (S E N W)
    neighb = np.full((_max_idx+1, 4),
                     # Se inicializa en -1 para los nodos sin vecinos
                     fill_value=-1)
    
    # Para cada elemento rectangular de `icone`
    for i in range(icone.shape[0]):
        p1, p2, p3, p4 = icone[i]
        
        # Esquina Sur-Oeste: p1 (icone[i][0])
        # Conexión en dirección Este [1]: p2 (icone[i][1])
        neighb[p1][1] = p2
        # Conexión en dirección Norte [2]: p4 (icone[i][3])
        neighb[p1][2] = p4
        
        # Esquina Sur-Este: p2 (icone[i][1])
        # Conexión en dirección Oeste [3]: p1 (icone[i][0])
        neighb[p2][3] = p1
        # Conexión en dirección Norte [2]: p3 (icone[i][2])
        neighb[p2][2] = p3
        
        # Esquina Norte-Este: p3 (icone[i][2])
        # Conexión en dirección Oeste [3]: p4 (icone[i][3])
        neighb[p3][3] = p4
        # Conexión en dirección Sur [0]: p2 (icone[i][1])
        neighb[p3][0] = p2
                
        # Esquina Norte-Oeste: p4 (icone[i][3])
        # Conexión en dirección Este [1]: p3 (icone[i][2])
        neighb[p4][1] = p3
        # Conexión en dirección Sur [0]: p1 (icone[i][0])
        neighb[p4][0] = p1

    return neighb

def fdm2d_dirichlet(K: np.ndarray, F: np.ndarray, DIR: np.ndarray):
    """
    **Descripción**: módulo para calcular y ensamblar las contribuciones de
    nodos pertenecientes a fronteras de tipo Dirichlet.
    
    **Entrada**:
    
    `K`: matriz del sistema (difusión + reacción)
    
    `F`: vector de flujo térmico
    
    `DIR`: matriz con la información sobre la frontera de tipo Dirchlet
        - Columna 1: número de nodo.
        - Columna 2: valor en ese nodo (escalar)
    
    **Salida**:
    
    `K`: matriz del sistema (difusión + reacción) luego de realizar las
    simplificaciones que surgen de aplicar la condición de borde Dirichlet.
    
    `F`: vector de flujo térmico luego de realizar las simplificaciones que
    surgen de aplicar la condición de borde Dirichlet.
    """
    # Índice de los nodos con condición Dirichlet
    inodes = DIR[:,0].astype(int)
    # transformar la fila en un vector de ceros con un 1 en la columna que
    # integra la diagonal principal de la matriz
    K[inodes, :] = 0.0          # Elementos fuera de la diagonal
    K[inodes, inodes] = 1.0     # Diagonal (nodos frontera)
    # Luego, siguiendo la notación introducida en (21), tendremos que:
    # $F_{i,j} = \bar\phi$
    F[inodes] = DIR[:, 1]       # Valor de la condición de borde
    
    return K, F

def fdm2d_neumann(K: np.ndarray, F: np.ndarray, xnode: np.ndarray, neighb: np.ndarray, NEU: np.ndarray):
    """
    **Descripción**: módulo para calcular y ensamblar las contribuciones de
    nodos pertenecientes a fronteras de tipo Neumann.
    
    **Entrada**:
    
    `F`: vector de flujo térmico.
    
    `xnode`: matriz de nodos con pares (x,y) representando las coordenadas de
    cada nodo de la malla.
    
    `neighb`: matriz de vecindad.
    
    `NEU`: matriz con la información sobre la frontera de tipo Neumann.
    - Columna 1: índice del nodo donde se aplica la condición de borde.
    - Columna 2: valor de flujo térmico (q) asociado al lado del elemento.
    - Columna 3: dirección y sentido del flujo:
        1. = Flujo en dirección eje-y, sentido negativo (S - South - Sur)
        2. = Flujo en dirección eje-x, sentido positivo (E - East - Este)
        3. = Flujo en dirección eje-y, sentido positivo (N - North - Norte)
        4. = Flujo en dirección eje-x, sentido negativo (W - West - Oeste)
    
    **Salida**:
    
    `F`: vector de flujo térmico con modificaciones luego de aplicar la condición de
    borde.
    """
    if NEU is None:
        return K, F
    
    # cantidad de nodos con condición Neumann
    M = len(NEU)
    
    for m in range(M):  # Para cada nodo
        # Índice nodo p
        p = int(NEU[m, 0])
        
        # Obtener la vecindad
        s, e, n, w = neighb[p]
        
        # Flujo impuesto
        q = NEU[m, 1]
        
        dx, dy = 0.0, 0.0
        
        if e == -1:     # Nodo en la frontera Este
            dx = np.abs(xnode[w, 0] - xnode[p, 0])
            
        if w == -1:     # Nodo en la frontera Oeste
            dx = np.abs(xnode[e, 0] - xnode[p, 0])
        
        if n == -1:     # Nodo en la frontera Norte
            dy = np.abs(xnode[s, 1] - xnode[p, 1])
        
        if s == -1:     # Nodo en la frontera SUR
            dy = np.abs(xnode[n, 1] - xnode[p, 1])
        
        norm = NEU[m, 2]
        if norm == 1 and s == -1:   # Frontera sur: Flujo en dirección eje-y, sentido negativo
            # print("Frontera sur - ", NEU[m])
            F[p] = F[p] - (2*q)/dy
        if norm == 2 and e == -1:   # Frontera este: Flujo en dirección eje-x, sentido positivo
            # print("Frontera este - ", NEU[m])
            F[p] = F[p] - (2*q)/dx
        if norm == 3 and n == -1:   # Frontera norte: Flujo en dirección eje-y, sentido positivo
            # print("Frontera norte - ", NEU[m])
            F[p] = F[p] - (2*q)/dy
        if norm == 4 and w == -1:   # Frontera oeste: Flujo en dirección eje-x, sentido negativo
            # print("Frontera oeste - ", NEU[m])
            F[p] = F[p] - (2*q)/dx
    
    return K, F

def fdm2d_robin(K: np.ndarray, F: np.ndarray, xnode: np.ndarray, neighb: np.ndarray, ROB: np.ndarray):
    """
    **Descripción**: módulo para calcular y ensamblar las contribuciones de
    nodos pertenecientes a fronteras de tipo Robin.
    
    **Entrada**:
    
    `K`: matriz del sistema (difusión + reacción)
    
    `F`: vector de flujo térmico.
    
    `xnode`: matriz de pares (x,y) representando las coordenadas de cada nodo de
    la malla.
    
    `neighb`: matriz de vecindad.
    
    `ROB`: matriz con la información sobre la frontera de tipo Robin.
    - Columna 1: índice del nodo donde se aplica la condición de borde.
    - Columna 2: valor de coeficiente de calor (h)
    - Columna 3: valor de temperatura de referencia (phi_inf).
    - Columna 3: dirección y sentido del flujo:
        1. = Flujo en dirección eje-y, sentido negativo (S - South - Sur)
        2. = Flujo en dirección eje-x, sentido positivo (E - East - Este)
        3. = Flujo en dirección eje-y, sentido positivo (N - North - Norte)
        4. = Flujo en dirección eje-x, sentido negativo (W - West - Oeste)
    
    **Salida**:
    `K`: matriz del sistema (difusión + reacción) con modificaciones luego
    de aplicar la condición de borde.
    
    `F`: vector de flujo térmico con modificaciones luego de aplicar la
    condición de borde.
    borde.
    """
    if ROB is None:
        return K, F
    
    # cantidad de nodos con condición Neumann
    M = len(ROB)
    
    for m in range(M):  # Para cada nodo
        # Índice nodo p
        p = int(ROB[m, 0])
        
        # Obtener la vecindad
        s, e, n, w = neighb[p]
        
        dx, dy = 0.0, 0.0
        
        if e == -1:     # Nodo en la frontera Este
            dx = np.abs(xnode[w, 0] - xnode[p, 0])
            
        if w == -1:     # Nodo en la frontera Oeste
            dx = np.abs(xnode[e, 0] - xnode[p, 0])
        
        if n == -1:     # Nodo en la frontera Norte
            dy = np.abs(xnode[s, 1] - xnode[p, 1])
        
        if s == -1:     # Nodo en la frontera SUR
            dy = np.abs(xnode[n, 1] - xnode[p, 1])
        
        # Coeficiente de calor
        h = ROB[m, 1]
        
        # Valor de temperatura de referencia
        phi_inf = ROB[m, 2]
        
        # Dirección y sentido del flujo
        norm = ROB[m, 3]
        
        if norm == 1 and s == -1:   # Frontera sur: Flujo en dirección eje-y, sentido negativo
            K[p, p] = K[p, p] + (2*h) / dy
            F[p] = F[p] + (2*h*phi_inf) / dy
        if norm == 2 and e == -1:   # Frontera este: Flujo en dirección eje-x, sentido positivo
            K[p, p] = K[p, p] + (2*h) / dx
            F[p] = F[p] + (2*h*phi_inf) / dx
        if norm == 3 and n == -1:   # Frontera norte: Flujo en dirección eje-y, sentido positivo
            K[p, p] = K[p, p] + (2*h) / dy
            F[p] = F[p] + (2*h*phi_inf) / dy
        if norm == 4 and w == -1:   # Frontera oeste: Flujo en dirección eje-x, sentido negativo
            K[p, p] = K[p, p] + (2*h) / dx
            F[p] = F[p] + (2*h*phi_inf) / dx
    
    return K, F


def fdm2d_gen_system(K: np.ndarray, F: np.ndarray, xnode: np.ndarray, neighb: np.ndarray, model: FDM_HeatModel):
    """
    **Descripción**: módulo para ensamblar los términos difusivo, reactivo y
    fuente de todos los nodos de la malla, generando el stencil adecuado 
    dependiendo de si es un nodo interior o de frontera
    
    **Entrada**:
    
    `K`: matriz del sistema (difusión + reacción)
    
    `F`: vector de flujo térmico.
    
    `xnode`: matriz de pares (x,y) representando cada nodo de la malla.
    
    `neighb`: matriz de vecindad.
    
    `model.k`: conductividad térmica del material. Es un vector que permite
    representar k(x,y).
    
    `model.c`: constante de reacción del material. Es un vector que permite
    representar c(x,y).
    
    `model.G`: fuente de calor. Es un vector que permite representar G(x,y).
    
    **Salida**:
    `K`: matriz del sistema (difusión + reacción) con modificaciones luego
    del ensamble.
    
    `F`: vector de flujo térmico con modificaciones luego del ensamble.
    """
    # cantidad de nodos de la malla
    N = len(xnode)
    
    for p in range(N):  # Para cada nodo
        # Obtener la vecindad
        s, e, n, w = neighb[p]
        
        # Distancia a cada vecino
        if e != -1:
            de = np.abs(xnode[e, 0] - xnode[p, 0])
        if n != -1:
            dn = np.abs(xnode[n, 1] - xnode[p, 1])
        if w != -1:
            dw = np.abs(xnode[w, 0] - xnode[p, 0])
        if s != -1:
            ds = np.abs(xnode[s, 1] - xnode[p, 1]) 
        
        # Coeficientes ecuación en diferencias Eje x
        if e == -1:     # Nodo en la frontera Este
            # ECUACION 38 / Documentación ProMetheus p.11
            cx = 0.0                      # Vecino i+1 (Este) -> Ficticio
                     # CHEQUEAR $\Delta x^2$ - No cabe otra posibilidad en realidad (no existe 'de')
            bx = -2.0 / (dw*dw)           # Nodo p (queda igual)
            ax = 2.0 / (dw*dw)            # Vecino i-1 (Oeste) -> Interior
            pass
            
        elif w == -1:   # Nodo en la frontera Oeste
            # ECUACION 38 / Documentación ProMetheus p.11
                     # CHEQUEAR $\Delta x^2$
            cx = 2.0 / (de*de)            # Vecino i+1 (Este) -> Interior
            bx = -2.0 / (de*de)           # Nodo p
            ax = 0.0                      # Vecino i-1 (Oeste) -> Ficticio
            pass
        
        else:           # Nodo interior
            # ECUACION 35 / Documentación ProMetheus p.10
            cx = 2.0 / (de*(de+dw))       # Vecino i+1 (Este) - Si la malla es uniforme queda 2 / (2*dx**2) = 1 / dx**2 
            bx = -2.0 / (de*dw)           # Nodo p
            ax = 2.0 / (dw*(de+dw))       # Vecino i-1 (Oeste)
        
        # Coeficientes ecuación en diferencias Eje x
        if n == -1:     # Nodo en la frontera norte
            # ECUACION 38 / Documentación ProMetheus p.11
            cy = 0.0                      # Vecino j+1 (Norte) -> Ficticio
                     # CHEQUEAR $\Delta x^2$
            by = -2.0 / (ds*ds)           # Nodo p
            ay = 2.0 / (ds*ds)            # Vecino j-1 (Sur) -> Interior
            pass
            
        elif s == -1:   # Nodo en la frontera sur
            # ECUACION 38 / Documentación ProMetheus p.11
                     # CHEQUEAR $\Delta x^2$
            cy = 2.0 / (dn*dn)            # Vecino j+1 (Norte) -> Interior
            by = -2.0 / (dn*dn)           # Nodo p
            ay = 0.0                      # Vecino j-1 (Sur) -> Ficticio
            pass
            
        else:           # Nodo interior
            # ECUACION 35 / Documentación ProMetheus p.10
            cy = 2.0 / (dn*(dn+ds))       # Vecino j+1 (Norte)
            by = -2.0 / (dn*ds)           # Nodo p
            ay = 2.0 / (ds*(dn+ds))       # Vecino j-1 (Sur)
        
        # Contribuciones presentes en cualquier nodo
        K[p, p] = model.c[p] - model.k[p] * bx - model.k[p] * by
        F[p] = model.G[p]   # Fuente
        
        if e != -1:     # Si p tiene un vecino al este
            # Sumar la contribución
            K[p, e] = -model.k[p] * cx
        if w != -1:     # Si p tiene un vecino al oeste
            K[p, w] = -model.k[p] * ax
        if n != -1:     # Si p tiene un vecino al norte
            K[p, n] = -model.k[p] * cy
        if s != -1:     # Si p tiene un vecino al sur
            K[p, s] = -model.k[p] * ay
        
    return K, F

def fdm2d_flux(PHI: np.ndarray, neighb: np.ndarray, xnode: np.ndarray, model_k: np.ndarray):
    """
    **Descripción**: módulo calcular el flujo de calor en todo el dominio. Se
    aplica la Ley de Fourier y se evalúa como fluye el calor en todos los puntos
    (nodos) del dominio.
    
    **Entrada**:
    
    `PHI`: vector solución. Cada elemento del vector representa un valor escalar
    asociado a cada nodo de la malla, y su posición dentro del vector depende de
    cómo se especificó cada nodo en xnode.
    
    `neighb`: matriz de vecindad.
        
    `xnode`: matriz de pares (x,y) representando cada nodo de la malla.
    
    `model.k`: conductividad térmica del material. Es un vector que permite
    representar k(x,y).
    
    **Salida**:
    
    `Q`: vector de flujo de calor. Para cada nodo se halla un vector
    bidimensional de flujo de calor, representado por un par (Qx,Qy)
    """
    
    # cantidad de nodos de la malla
    N = len(xnode)
    
    # Vector de flujo de calor
    Q = np.zeros((N, 2))    # [Qx, Qy]
    
    for p in range(N):  # Para cada nodo
        # Obtener la vecindad
        s, e, n, w = neighb[p]
        
        if e == -1:       # Nodo en la frontera Este
            dx_w = np.abs(xnode[p, 0] - xnode[w, 0])
            dphi_dx = (PHI[p] - PHI[w]) / dx_w

        elif w == -1:       # Nodo en la frontera Oeste
            dx_e = np.abs(xnode[e, 0] - xnode[p, 0])
            dphi_dx = (PHI[e] - PHI[p]) / dx_e

        else:     # Nodo interior para dirección x
            dx_e = np.abs(xnode[e, 0] - xnode[p, 0])
            dx_w = np.abs(xnode[p, 0] - xnode[w, 0])
            dphi_dx = (PHI[e] - PHI[w]) / (dx_e + dx_w)
        
        if n == -1:       # Nodo en la frontera norte
            dy_s = np.abs(xnode[p, 1] - xnode[s, 1])
            dphi_dy = (PHI[p] - PHI[s]) / dy_s
        
        elif s == -1:       # Nodo en la frontera sur
            dy_n = np.abs(xnode[n, 1] - xnode[p, 1])
            dphi_dy = (PHI[n] - PHI[p]) / dy_n
            
        else:     # Nodo interior para dirección y
            dy_n = np.abs(xnode[n, 1] - xnode[p, 1])
            dy_s = np.abs(xnode[p, 1] - xnode[s, 1])
            dphi_dy = (PHI[n] - PHI[s]) / (dy_n + dy_s)
        
        Q[p, 0] = -model_k[p] * dphi_dx
        Q[p, 1] = -model_k[p] * dphi_dy

    return Q

def fdm2d_explicit_delta_t(xnode: np.ndarray, model: FDM_HeatModel):
    """
    **Descripción**: módulo para calcular el paso temporal crítico para esquema
    temporal explícito a partir de las constantes del modelo y las dimensiones
    de los elementos de la malla.
    
    **Entrada**:
    
    `xnode`: matriz de pares (x,y) representando cada nodo de la malla.
    
    `model`: struct con todos los datos del modelo (constantes, esquema
    numérico, etc.).
    
    **Salida**:
    
    `dt`: paso temporal crítico para método explícito.
    """
    # Encontrar el k más grande (conductividad) que maximiza kappa (difusividad)
    # que minimiza el dt
    max_k = np.max(model.k)
    kappa = max_k / (model.rho * model.cp)
    
    # Lambda para cumplir la desigualdad (1-2*lambda) <= 0 con lambda = kappa * dt / h
    lambda_ = 0.25 # 1/4
    
    # Encontrar el h más chico que minimiza dt
    # @TODO Generalizar: Esta formulación es específica del orden de xnode (X-Y)
    h = np.min([
        np.min(np.abs(np.diff(xnode[:,0]))),    
        np.max(np.abs(np.diff(xnode[:,1])))
    ])
    # h = xnode[1][0] - xnode[0][0]
    
    dt = (lambda_ * h**2) / kappa
    
    return dt

def fdm2d_explicit(K: np.ndarray, F: np.ndarray, xnode: np.ndarray, neighb: np.ndarray, model: FDM_HeatModel, dt: float):
    """
    **Descripción**: módulo para resolver el sistema lineal de ecuaciones
    utilizando esquema temporal explícito. El primer valor (primer columna) es
    la condición inicial.
    
    **Entrada**:
    
    `K`: matriz del sistema (difusión + reacción)
    
    `F`: vector de flujo térmico.
    
    `xnode`: matriz de nodos con pares (x,y) representando las coordenadas de
    cada nodo de la malla.
    
    `neighb`: matriz de vecindad.
    
    `model`: struct con todos los datos del modelo (constantes, esquema
    numérico, etc.)
    
    `dt`: paso temporal crítico para método explícito.
    
    **Salida**:
    
    `PHI`: matriz solución. Cada elemento del vector representa un valor escalar
    asociado a cada nodo de la malla, y su posición dentro del vector depende de
    cómo se especificó cada nodo en xnode. Cada columna representa una iteración
    del esquema temporal (en total nit columnas).
    
    `Q`: matriz de flujo de calor. Para cada nodo se halla un vector
    bidimensional de flujo de calor, representado por un par (Qx,Qy). Cada par
    de columnas representa una iteración del esquema temporal (en total 2*nit
    columnas).
    """
    print("Explicit temporal scheme")
    print(f"dt = {dt}")
    
    # Solución t = 0
    PHI = [model.PHI_n]
    Q = [fdm2d_flux(PHI[-1], neighb, xnode, model.k)]
    
    _a = (dt / (model.rho * model.cp)) * F
    _b = np.identity(K.shape[0]) - (dt / (model.rho * model.cp)) * K
    
    for i in range(1, model.maxit):
        PHI.append(_a + _b @ PHI[i-1])
        Q.append(fdm2d_flux(PHI[-1], neighb, xnode, model.k))
        
        # Error relativo
        if (np.linalg.norm(PHI[i]-PHI[i-1], ord=2) / np.linalg.norm(PHI[i], ord=2)) < model.tol:
            print("Stop on iter ", i)
            break
        
        if i % 100 == 0:
            print(f"Iter {i}")
    
    return PHI, Q

def fdm2d_implicit(K: np.ndarray, F: np.ndarray, xnode: np.ndarray, neighb: np.ndarray, model: FDM_HeatModel, dt: float):
    """
    **Descripción**: módulo para resolver el sistema lineal de ecuaciones
    utilizando esquema temporal implícito. El primer valor (primer columna) es
    la condición inicial.
    
    **Entrada**:
    
    `K`: matriz del sistema (difusión + reacción)
    
    `F`: vector de flujo térmico.
    
    `xnode`: matriz de nodos con pares (x,y) representando las coordenadas de
    cada nodo de la malla.
    
    `neighb`: matriz de vecindad.
    
    `model`: struct con todos los datos del modelo (constantes, esquema
    numérico, etc.)
    
    `dt`: paso temporal arbitrario para método explícito.
    
    **Salida**:
    
    `PHI`: matriz solución. Cada elemento del vector representa un valor escalar
    asociado a cada nodo de la malla, y su posición dentro del vector depende de
    cómo se especificó cada nodo en xnode. Cada columna representa una iteración
    del esquema temporal (en total nit columnas).
    
    `Q`: matriz de flujo de calor. Para cada nodo se halla un vector
    bidimensional de flujo de calor, representado por un par (Qx,Qy). Cada par
    de columnas representa una iteración del esquema temporal (en total 2*nit
    columnas).
    """
    print("Implicit temporal scheme")
    print(f"dt = {dt}")
    
    # Solución t = 0
    PHI = [model.PHI_n]
    Q = [fdm2d_flux(PHI[-1], neighb, xnode, model.k)]
    
    _a = (model.rho * model.cp) / dt
    _b = np.identity(K.shape[0]) * _a + K
    
    for i in range(1, model.maxit):
        PHI.append(
            np.linalg.solve(_b, (F + _a * PHI[i-1]))
        )
        Q.append(fdm2d_flux(PHI[-1], neighb, xnode, model.k))
        
        # Error relativo
        if (np.linalg.norm(PHI[i]-PHI[i-1], ord=2) / np.linalg.norm(PHI[i], ord=2)) < model.tol:
            print("Stop on iter ", i)
            break
        
        if i % 100 == 0:
            print(f"Iter {i}")
    
    return PHI, Q

def fdm2d_semiimplicit(K: np.ndarray, F: np.ndarray, xnode: np.ndarray, neighb: np.ndarray, model: FDM_HeatModel, dt: float):
    """
    **Descripción**: módulo para resolver el sistema lineal de ecuaciones
    utilizando esquema temporal semi-implícito. El primer valor (primer columna)
    es la condición inicial.
    
    **Entrada**:
    
    `K`: matriz del sistema (difusión + reacción)
    
    `F`: vector de flujo térmico.
    
    `xnode`: matriz de nodos con pares (x,y) representando las coordenadas de
    cada nodo de la malla.
    
    `neighb`: matriz de vecindad.
    
    `model`: struct con todos los datos del modelo (constantes, esquema
    numérico, etc.)
    
    `dt`: paso temporal arbitrario para método explícito.
    
    **Salida**:
    
    `PHI`: matriz solución. Cada elemento del vector representa un valor escalar
    asociado a cada nodo de la malla, y su posición dentro del vector depende de
    cómo se especificó cada nodo en xnode. Cada columna representa una iteración
    del esquema temporal (en total nit columnas).
    
    `Q`: matriz de flujo de calor. Para cada nodo se halla un vector
    bidimensional de flujo de calor, representado por un par (Qx,Qy). Cada par
    de columnas representa una iteración del esquema temporal (en total 2*nit
    columnas).
    """
    print("Semi-Implicit temporal scheme")
    print(f"dt = {dt}")
    
    # Solución t = 0
    PHI = [model.PHI_n]
    Q = [fdm2d_flux(PHI[-1], neighb, xnode, model.k)]
    
    _a = (model.rho * model.cp) / dt
    _b = np.identity(K.shape[0]) * _a + 0.5 * K
    _c = np.identity(K.shape[0]) * _a - 0.5 * K
    
    for i in range(1, model.maxit):
        PHI.append(
            np.linalg.solve(_b, (F + _c @ PHI[i-1]))
        )
        Q.append(fdm2d_flux(PHI[-1], neighb, xnode, model.k))
        
        # Error relativo
        if (np.linalg.norm(PHI[i]-PHI[i-1], ord=2) / np.linalg.norm(PHI[i], ord=2)) < model.tol:
            print("Stop on iter ", i)
            break
        
        if i % 100 == 0:
            print(f"Iter {i}")
    
    return PHI, Q

def fdm2d_solve(K: np.ndarray, F: np.ndarray, xnode: np.ndarray, neighb: np.ndarray, model: FDM_HeatModel):
    """
    **Descripción**: módulo para resolver el sistema lineal de ecuaciones. En
    este módulo se realizan los cálculos para obtener la solución propia del
    método numérico. Dicha solución se obtiene por dos vías:
    - Sin la aplicación de esquemas temporales, es decir, la solución del
    sistema en estado estacionario. Resolución por método directo.
    - Aplicación de esquemas temporales, a saber: método explícito y método
    implícito. Se evalúa la evolución temporal del sistema desde un estado
    inicial conocido hasta un determinado instante de tiempo. Resolución por
    método iterativo.
    
    **Entrada**:
    
    `K`: matriz del sistema (difusión + reacción)
    
    `F`: vector de flujo térmico.
    
    `xnode`: matriz de nodos con pares (x,y) representando las coordenadas de
    cada nodo de la malla.
    
    `neighb`: matriz de vecindad.
    
    `model`: struct con todos los datos del modelo (constantes, esquema
    numérico, etc.).
    
    **Salida**:
    
    `PHI`: matriz solución. Cada elemento del vector representa un valor escalar
    asociado a cada nodo de la malla, y su posición dentro del vector depende de
    cómo se especificó cada nodo en xnode. Cada columna representa una iteración
    del esquema temporal (en total nit columnas).
    
    `Q`: matriz de flujo de calor. Para cada nodo se halla un vector
    bidimensional de flujo de calor, representado por un par (Qx,Qy). Cada par
    de columnas representa una iteración del esquema temporal (en total
    2*nit columnas).
    """
    # Esquema temporal: [0] Explícito, [1] Implícito, [X] Estacionario
     
    if model.ts == 0:       # Explícito - Forward Euler
        # Paso temporal del método explícito (Forward Euler)
        dt = fdm2d_explicit_delta_t(xnode, model)
        
        PHI, Q = fdm2d_explicit(K, F, xnode, neighb, model, dt)

    elif model.ts == 1:     # Implícito - Backward Euler
        # Paso temporal arbitrario
        dt = model.dt
        
        PHI, Q = fdm2d_implicit(K, F, xnode, neighb, model, dt)

    elif model.ts == 2:      # Semi-implícito - Crank-Nicholson
        # Paso temporal arbitrario
        dt = model.dt
        
        PHI, Q = fdm2d_semiimplicit(K, F, xnode, neighb, model, dt)

    else:                   # Estado estacionario
        # Resolución del sistema lineal de ecuaciones
        PHI = np.linalg.solve(K, F)
        
        # Cálculo del flujo de calor
        Q = fdm2d_flux(PHI, neighb, xnode, model.k)
    
    return PHI, Q
    


def fdm2d(xnode: np.ndarray, icone: np.ndarray, DIR: np.ndarray, NEU: np.ndarray, ROB: np.ndarray, model: FDM_HeatModel):
    """
    - `xnode`: matriz de pares (x,y) representando cada nodo de la malla    
    - `icone`: matriz de conectividad. Todos los nodos se conectan formando
    elementos rectangulares (pero el tratamiento general del método es por nodos)
    - `DIR`: matriz de nodos frontera tipo Dirichlet.
    - `NEU`: matriz de pares de nodos frontera tipo Neumann.
    - `ROB`: matriz de pares de nodos frontera tipo Robin.
    - `model`: struct con todos los datos del modelo (constantes, esquema numérico, etc.)
    """
    # 1. Inicialización de variables principales del sistema
    K, F = fdm2d_initialize(model.nnodes)
    
    # 2. Armado de la matriz de vecindad
    neighb = fdm2d_neighbors(icone)
    
    # 3. Ensamble de coeficientes del sistema
    K, F = fdm2d_gen_system(K, F, xnode, neighb, model)
    
    # 4. Ensamble de nodos frontera Neumann
    K, F = fdm2d_neumann(K, F, xnode, neighb, NEU)
    
    # 5. Ensamble de nodos frontera Robin
    K, F = fdm2d_robin(K, F, xnode, neighb, ROB)
    
    # # 6. Ensamble de nodos frontera Dirichlet
    K, F = fdm2d_dirichlet(K, F, DIR)
    
    # 7. Resolución del sistema lineal de ecuaciones
    PHI, Q = fdm2d_solve(K, F, xnode, neighb, model)

    return PHI, Q