import numpy as np
import matplotlib.pyplot as plt

def validar_poligono_ortogonal(vertices, tol=1e-12):
    """
    Valida un polígono plano, simple y ortogonal.

    Parámetros
    ----------
    vertices : array_like, shape (n, 2)
        Coordenadas [x, y] de los vértices, en orden.
        No es necesario repetir el primer vértice al final.

    tol : float
        Tolerancia numérica para comparar coordenadas.

    Retorna
    -------
    vertices : np.ndarray
        Vértices convertidos a float y normalizados.

    Raises
    ------
    ValueError
        Si la geometría no cumple alguna de las condiciones requeridas.

    Notas
    -----
    Se exige:
        - al menos 4 vértices
        - coordenadas finitas
        - ningún segmento de longitud cero
        - todos los segmentos horizontales o verticales
        - polígono simple, sin auto-intersecciones
        - orientación antihoraria
    """

    # ---------------------------------------------------------
    # 1. Conversión y forma
    # ---------------------------------------------------------
    vertices = np.asarray(vertices, dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(
            "Los vértices deben tener forma (n, 2)."
        )

    if len(vertices) < 4:
        raise ValueError(
            "Un polígono debe tener al menos 4 vértices."
        )

    if not np.all(np.isfinite(vertices)):
        raise ValueError(
            "Las coordenadas deben ser valores finitos."
        )

    # ---------------------------------------------------------
    # 2. Si se repite el primer vértice al final, eliminarlo
    # ---------------------------------------------------------
    if np.allclose(vertices[0], vertices[-1], atol=tol, rtol=0):
        vertices = vertices[:-1]

    if len(vertices) < 4:
        raise ValueError(
            "El polígono debe tener al menos 4 vértices distintos."
        )

    # ---------------------------------------------------------
    # 3. Segmentos consecutivos, incluyendo el cierre
    # ---------------------------------------------------------
    p0 = vertices
    p1 = np.roll(vertices, -1, axis=0)

    dx = p1[:, 0] - p0[:, 0]
    dy = p1[:, 1] - p0[:, 1]

    # ---------------------------------------------------------
    # 4. Segmentos de longitud cero
    # ---------------------------------------------------------
    if np.any(
        (np.abs(dx) <= tol) &
        (np.abs(dy) <= tol)
    ):
        i = np.where(
            (np.abs(dx) <= tol) &
            (np.abs(dy) <= tol)
        )[0][0]

        raise ValueError(
            f"El segmento {i} tiene longitud cero."
        )

    # ---------------------------------------------------------
    # 5. Ortogonalidad
    # ---------------------------------------------------------
    diagonales = (
        (np.abs(dx) > tol) &
        (np.abs(dy) > tol)
    )

    if np.any(diagonales):
        i = np.where(diagonales)[0][0]

        raise ValueError(
            f"El segmento {i} no es horizontal ni vertical: "
            f"{vertices[i]} -> {p1[i]}"
        )

    # ---------------------------------------------------------
    # 6. Área firmada
    #
    #    > 0  -> antihorario
    #    < 0  -> horario
    # ---------------------------------------------------------
    area2 = np.sum(
        vertices[:, 0] * p1[:, 1]
        - p1[:, 0] * vertices[:, 1]
    )

    if area2 <= tol:
        if abs(area2) <= tol:
            raise ValueError(
                "El polígono tiene área nula."
            )
        else:
            raise ValueError(
                "Los vértices deben estar ordenados "
                "en sentido antihorario."
            )

    # ---------------------------------------------------------
    # 7. Intersecciones entre segmentos no consecutivos
    #
    #    Para un polígono ortogonal podemos comprobar esto
    #    de manera sencilla.
    # ---------------------------------------------------------
    n = len(vertices)

    def entre(a, b, x):
        return (
            min(a, b) - tol <= x <= max(a, b) + tol
        )

    for i in range(n):
        a0 = vertices[i]
        a1 = vertices[(i + 1) % n]

        a_horizontal = abs(a0[1] - a1[1]) <= tol

        for j in range(i + 1, n):
            # Segmentos consecutivos comparten vértice
            if j == i + 1:
                continue

            # Primer y último segmento también son consecutivos
            if i == 0 and j == n - 1:
                continue

            b0 = vertices[j]
            b1 = vertices[(j + 1) % n]

            b_horizontal = abs(b0[1] - b1[1]) <= tol

            if a_horizontal and b_horizontal:
                # Paralelos horizontales
                if abs(a0[1] - b0[1]) <= tol:
                    if (
                        max(min(a0[0], a1[0]), min(b0[0], b1[0]))
                        <=
                        min(max(a0[0], a1[0]), max(b0[0], b1[0]))
                        + tol
                    ):
                        raise ValueError(
                            f"Los segmentos {i} y {j} "
                            "se intersectan o se superponen."
                        )

            elif (not a_horizontal) and (not b_horizontal):
                # Paralelos verticales
                if abs(a0[0] - b0[0]) <= tol:
                    if (
                        max(min(a0[1], a1[1]), min(b0[1], b1[1]))
                        <=
                        min(max(a0[1], a1[1]), max(b0[1], b1[1]))
                        + tol
                    ):
                        raise ValueError(
                            f"Los segmentos {i} y {j} "
                            "se intersectan o se superponen."
                        )

            else:
                # Uno horizontal y otro vertical
                if a_horizontal:
                    h0, h1 = a0, a1
                    v0, v1 = b0, b1
                else:
                    h0, h1 = b0, b1
                    v0, v1 = a0, a1

                x = v0[0]
                y = h0[1]

                if (
                    entre(h0[0], h1[0], x)
                    and entre(v0[1], v1[1], y)
                ):
                    raise ValueError(
                        f"Los segmentos {i} y {j} "
                        "se intersectan."
                    )

    return vertices

def extraer_coordenadas(vertices):
    """
    Extrae las coordenadas x e y que aparecen en los vértices
    del polígono.

    Retorna
    -------
    x_geom : np.ndarray
        Coordenadas x únicas y ordenadas.
    y_geom : np.ndarray
        Coordenadas y únicas y ordenadas.
    """

    vertices = np.asarray(vertices, dtype=float)

    x_geom = np.unique(vertices[:, 0])
    y_geom = np.unique(vertices[:, 1])

    return x_geom, y_geom

def construir_eje(coordenadas, niveles):
    """
    Construye un eje subdividiendo cada intervalo entre coordenadas
    consecutivas.

    Parámetros
    ----------
    coordenadas : array-like
        Coordenadas de interés que deben pertenecer al eje.
        Deben estar ordenadas y no repetidas.

    niveles : int
        Nivel de subdivisión.
        Cada intervalo se divide en 2**niveles subintervalos.

        niveles = 0 -> no subdivide
        niveles = 1 -> 2 subintervalos
        niveles = 2 -> 4 subintervalos
        niveles = 3 -> 8 subintervalos
        ...

    Retorna
    -------
    eje : ndarray
        Coordenadas del eje, incluyendo todas las coordenadas
        originales y las nuevas coordenadas intermedias.
    """

    coordenadas = np.asarray(coordenadas, dtype=float)

    # =========================================================
    # Validación
    # =========================================================

    if coordenadas.ndim != 1:
        raise ValueError(
            "coordenadas debe ser un vector unidimensional."
        )

    if len(coordenadas) == 0:
        raise ValueError(
            "coordenadas no puede estar vacío."
        )

    if not np.all(np.isfinite(coordenadas)):
        raise ValueError(
            "coordenadas debe contener valores finitos."
        )

    if np.any(np.diff(coordenadas) <= 0):
        raise ValueError(
            "coordenadas debe estar ordenado estrictamente "
            "de menor a mayor."
        )

    if not isinstance(niveles, (int, np.integer)):
        raise ValueError(
            "niveles debe ser un entero."
        )

    if niveles < 0:
        raise ValueError(
            "niveles debe ser mayor o igual que cero."
        )

    # =========================================================
    # Número de subdivisiones
    # =========================================================

    nsub = 2**niveles

    # =========================================================
    # Construcción del eje
    # =========================================================

    eje = []

    for i in range(len(coordenadas) - 1):

        x0 = coordenadas[i]
        x1 = coordenadas[i + 1]

        # Incluye x0 pero no x1.
        # x1 será incluido al comenzar el siguiente intervalo,
        # o explícitamente al final.
        intervalo = np.linspace(
            x0,
            x1,
            nsub + 1
        )

        eje.extend(intervalo[:-1])

    # Agregar la última coordenada
    eje.append(coordenadas[-1])

    return np.asarray(eje)

def generar_grilla(xx, yy):
    """
    Genera una grilla cartesiana a partir de los ejes xx e yy.

    La numeración es:
        x creciente primero,
        luego y creciente.

    Retorna
    -------
    X, Y : np.ndarray
        Matrices de coordenadas.
    xnode : np.ndarray
        Matriz (nnodes, 2) con las coordenadas de los nodos.
    """

    X, Y = np.meshgrid(
        xx,
        yy,
        indexing="xy"
    )

    xnode = np.column_stack(
        (X.ravel(), Y.ravel())
    )

    return X, Y, xnode

def punto_en_poligono(x, y, vertices):
    """
    Determina si un punto está dentro de un polígono.

    Utiliza el algoritmo de ray casting.

    Parámetros
    ----------
    x, y : float
        Coordenadas del punto.

    vertices : np.ndarray, shape (n, 2)
        Vértices del polígono.

    Retorna
    -------
    bool
        True si el punto está dentro del polígono.
    """

    dentro = False

    n = len(vertices)

    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]

        # ¿El segmento cruza horizontalmente el nivel y?
        cruza = (y0 > y) != (y1 > y)

        if cruza:
            # Coordenada x donde el segmento intersecta y = constante
            x_interseccion = (
                x0
                + (y - y0) * (x1 - x0) / (y1 - y0)
            )

            if x < x_interseccion:
                dentro = not dentro

    return dentro

def punto_en_dominio(x, y, exterior, huecos = None):
    """
    Determina si un punto pertenece al dominio.

    El dominio está definido por un polígono exterior
    menos uno o más huecos.
    """

    if not punto_en_poligono(x, y, exterior):
        return False

    if huecos:
        for hueco in huecos:
            if punto_en_poligono(x, y, hueco):
                return False

    return True

def recortar_grilla(poligonos, xx, yy):
    """
    Recorta una grilla cartesiana al dominio definido por un
    polígono exterior y uno o más huecos.

    Parámetros
    ----------
    poligonos : list of np.ndarray
        Lista de polígonos.
        poligonos[0] corresponde al exterior.
        poligonos[1:] corresponden a los huecos.

    xx : np.ndarray
        Coordenadas x de la grilla.

    yy : np.ndarray
        Coordenadas y de la grilla.

    Retorna
    -------
    xnode : np.ndarray, shape (nnodes, 2)
        Coordenadas de los nodos pertenecientes al dominio.

    icone : np.ndarray, shape (nelements, 4)
        Conectividad de los elementos cuadriláteros.

        Orden:
            inferior izquierda
            inferior derecha
            superior derecha
            superior izquierda

    P : np.ndarray
        Matriz auxiliar de índices de nodos.
        Los puntos que no pertenecen a la malla tienen valor -1.
    """

    # ---------------------------------------------------------
    # 1. Separar exterior y huecos
    # ---------------------------------------------------------

    if len(poligonos) == 0:
        raise ValueError(
            "No se proporcionaron polígonos."
        )

    exterior = poligonos[0]
    
    if len(poligonos) > 1:    
        huecos = poligonos[1:]
    else:
        huecos = None

    xx = np.asarray(xx, dtype=float)
    yy = np.asarray(yy, dtype=float)

    if len(xx) < 2 or len(yy) < 2:
        raise ValueError(
            "xx e yy deben contener al menos dos coordenadas."
        )

    # ---------------------------------------------------------
    # 2. Comprobar que los vértices de todos los polígonos
    #    están presentes en la grilla
    # ---------------------------------------------------------

    for poligono in poligonos:

        for x, y in poligono:

            if not np.any(np.isclose(xx, x)):
                raise ValueError(
                    f"La coordenada x={x} del polígono "
                    "no está presente en xx."
                )

            if not np.any(np.isclose(yy, y)):
                raise ValueError(
                    f"La coordenada y={y} del polígono "
                    "no está presente en yy."
                )

    # ---------------------------------------------------------
    # 3. Grilla cartesiana auxiliar completa
    # ---------------------------------------------------------

    N = len(xx)
    M = len(yy)

    P_full = (
        np.arange(N)[None, :]
        + np.arange(M)[:, None] * N
    )

    # ---------------------------------------------------------
    # 4. Determinar qué celdas pertenecen al dominio
    # ---------------------------------------------------------

    elementos_full = []

    for j in range(M - 1):

        for i in range(N - 1):

            # Centro de la celda
            xc = 0.5 * (xx[i] + xx[i + 1])
            yc = 0.5 * (yy[j] + yy[j + 1])

            # IMPLEMENTACION VIEJA SIN HUECO
            # if punto_en_poligono(xc, yc, vertices):
            if punto_en_dominio(xc, yc, exterior, huecos):
                bl = P_full[j,     i]
                br = P_full[j,     i + 1]
                tr = P_full[j + 1, i + 1]
                tl = P_full[j + 1, i]

                elementos_full.append([
                    bl,
                    br,
                    tr,
                    tl
                ])

    if not elementos_full:
        raise ValueError(
            "La grilla no contiene ninguna celda dentro "
            "del dominio."
        )

    elementos_full = np.asarray(
        elementos_full,
        dtype=int
    )

    # ---------------------------------------------------------
    # 5. Obtener solamente los nodos utilizados
    # ---------------------------------------------------------

    nodos_utilizados = np.unique(
        elementos_full.ravel()
    )

    # ---------------------------------------------------------
    # 6. Renumerar nodos
    # ---------------------------------------------------------

    # Mapeo:
    #
    # índice antiguo -> índice nuevo
    #
    # Ejemplo:
    #   0  -> 0
    #   1  -> 1
    #   2  -> -1
    #   3  -> 2
    #
    mapa = np.full(
        N * M,
        -1,
        dtype=int
    )

    mapa[nodos_utilizados] = np.arange(
        len(nodos_utilizados)
    )

    icone = mapa[elementos_full]

    # ---------------------------------------------------------
    # 7. Coordenadas de los nodos conservados
    # ---------------------------------------------------------

    X_full, Y_full = np.meshgrid(
        xx,
        yy,
        indexing="xy"
    )

    x_full = np.column_stack([
        X_full.ravel(),
        Y_full.ravel()
    ])

    xnode = x_full[nodos_utilizados]

    # ---------------------------------------------------------
    # 8. Matriz P renumerada
    #
    #    -1 = punto que no pertenece a la malla
    # ---------------------------------------------------------

    P = mapa.reshape(M, N)

    return xnode, icone, P

def obtener_nodos_frontera(poligonos, xnode, tol=1e-12):
    """
    Obtiene los nodos de xnode pertenecientes a cada frontera
    de cada polígono.

    El primer polígono corresponde al exterior y los siguientes
    corresponden a huecos.

    Parámetros
    ----------
    poligonos : list of np.ndarray
        Lista de polígonos.

    xnode : np.ndarray, shape (nnodes, 2)
        Coordenadas de los nodos de la malla.

    tol : float
        Tolerancia numérica.

    Retorna
    -------
    fronteras : list of list of np.ndarray
        fronteras[p][i] contiene los índices de xnode que
        pertenecen al lado i del polígono p.

        fronteras[0] corresponde al exterior.
        fronteras[1:] corresponden a los huecos.
    """

    xnode = np.asarray(xnode, dtype=float)

    if xnode.ndim != 2 or xnode.shape[1] != 2:
        raise ValueError(
            "xnode debe tener forma (nnodes, 2)."
        )

    if len(poligonos) == 0:
        raise ValueError(
            "No se proporcionaron polígonos."
        )

    fronteras = []
    
    # =========================================================
    # Recorrer todos los polígonos
    # =========================================================

    for poligono in poligonos:
        
        poligono = validar_poligono_ortogonal(
            poligono,
            tol=tol
        )

        n = len(poligono)

        fronteras_poligono = []
        
        # =====================================================
        # Recorrer los lados del polígono
        # =====================================================

        for i in range(n):

            # -----------------------------------------------------
            # Segmento de frontera
            # -----------------------------------------------------

            p0 = poligono[i]
            p1 = poligono[(i + 1) % n]

            x0, y0 = p0
            x1, y1 = p1

            # -----------------------------------------------------
            # Determinar si es horizontal o vertical
            # -----------------------------------------------------

            horizontal = abs(y1 - y0) <= tol

            if horizontal:

                y = y0
                xmin = min(x0, x1)
                xmax = max(x0, x1)

                mascara = (
                    (np.abs(xnode[:, 1] - y) <= tol)
                    &
                    (xnode[:, 0] >= xmin - tol)
                    &
                    (xnode[:, 0] <= xmax + tol)
                )

                indices = np.where(mascara)[0]

                # Ordenar siguiendo p0 -> p1
                if x1 >= x0:
                    indices = indices[
                        np.argsort(xnode[indices, 0])
                    ]
                else:
                    indices = indices[
                        np.argsort(-xnode[indices, 0])
                    ]

            else:

                x = x0
                ymin = min(y0, y1)
                ymax = max(y0, y1)

                mascara = (
                    (np.abs(xnode[:, 0] - x) <= tol)
                    &
                    (xnode[:, 1] >= ymin - tol)
                    &
                    (xnode[:, 1] <= ymax + tol)
                )

                indices = np.where(mascara)[0]

                # Ordenar siguiendo p0 -> p1
                if y1 >= y0:
                    indices = indices[
                        np.argsort(xnode[indices, 1])
                    ]
                else:
                    indices = indices[
                        np.argsort(-xnode[indices, 1])
                    ]
            
            # -------------------------------------------------
            # Comprobar que se encontraron nodos
            # -------------------------------------------------

            if len(indices) == 0:
                raise ValueError(
                    f"No se encontraron nodos para la frontera "
                    f"{i}: {p0} -> {p1}."
                )

            fronteras_poligono.append(indices)
        
        # -----------------------------------------------------
        # Guardar las fronteras de este polígono
        # -----------------------------------------------------

        fronteras.append(fronteras_poligono)

    return fronteras

def separar_poligonos(vertices, tol=1e-12):
    """
    Separa una secuencia de vértices en polígonos cerrados.

    Cada polígono debe cerrarse repitiendo su primer vértice.
    El primer polígono corresponde al exterior y los siguientes
    corresponden a huecos.

    Parámetros
    ----------
    vertices : array_like, shape (n, 2)
        Secuencia de vértices de todos los polígonos.
        Cada polígono debe terminar repitiendo su primer vértice.

    tol : float
        Tolerancia numérica para detectar la repetición del
        primer vértice.

    Retorna
    -------
    poligonos : list of np.ndarray
        Lista de polígonos validados.

        poligonos[0] : polígono exterior
        poligonos[1:] : huecos

    Raises
    ------
    ValueError
        Si algún polígono no está correctamente cerrado o no
        cumple las condiciones geométricas.
    """

    vertices = np.asarray(vertices, dtype=float)

    # ---------------------------------------------------------
    # 1. Conversión y forma
    # ---------------------------------------------------------

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(
            "Los vértices deben tener forma (n, 2)."
        )

    if len(vertices) == 0:
        raise ValueError(
            "No se proporcionaron vértices."
        )

    if not np.all(np.isfinite(vertices)):
        raise ValueError(
            "Las coordenadas deben ser valores finitos."
        )

    # ---------------------------------------------------------
    # 2. Separar polígonos
    # ---------------------------------------------------------

    poligonos = []
    poligono_actual = []

    for i, punto in enumerate(vertices):

        poligono_actual.append(punto)

        # Todavía no podemos cerrar un polígono con un único punto
        if len(poligono_actual) < 2:
            continue

        # ¿El punto actual coincide con el primero?
        if np.allclose(
            punto,
            poligono_actual[0],
            atol=tol,
            rtol=0
        ):

            # El último punto es una repetición del primero,
            # por lo tanto no forma parte del polígono almacenado.
            poligono = np.asarray(
                poligono_actual[:-1],
                dtype=float
            )

            # -------------------------------------------------
            # 3. Validar inmediatamente el polígono
            # -------------------------------------------------

            poligono = validar_poligono_ortogonal(
                poligono,
                tol=tol
            )

            poligonos.append(poligono)

            # Comenzar un nuevo polígono
            poligono_actual = []

    # ---------------------------------------------------------
    # 4. Comprobar que no quedó un polígono abierto
    # ---------------------------------------------------------

    if len(poligono_actual) > 0:
        raise ValueError(
            "La secuencia de vértices termina sin cerrar "
            "el último polígono. Cada polígono debe repetir "
            "su primer vértice al final."
        )

    # ---------------------------------------------------------
    # 5. Debe existir al menos un polígono
    # ---------------------------------------------------------

    if len(poligonos) == 0:
        raise ValueError(
            "No se encontró ningún polígono cerrado."
        )

    return poligonos












def dibujar_malla(
    xnode,
    icone,
    poligonos=None,
    fronteras=None,
    mostrar_numeros=False,
    mostrar_nodos=True,
    mostrar_elementos=False,
    mostrar_fronteras=True,
):
    """
    Grafica una malla de elementos cuadriláteros.

    Parámetros
    ----------
    xnode : ndarray, shape (nnodes, 2)
        Coordenadas [x, y] de los nodos.

    icone : ndarray, shape (nelem, 4)
        Conectividad de los elementos cuadriláteros.

        Cada fila contiene:
            [inferior_izquierdo,
             inferior_derecho,
             superior_derecho,
             superior_izquierdo]

    poligonos : list of ndarray, opcional
        Lista de polígonos que define la geometría original.

        poligonos[0] corresponde al contorno exterior.
        poligonos[1:] corresponden a los agujeros.

        Cada polígono debe tener forma (n, 2) y no repetir
        el primer vértice al final.

    fronteras : list of list of ndarray, opcional
        Lista anidada de nodos pertenecientes a cada frontera.

        fronteras[0] contiene las fronteras del polígono exterior.
        fronteras[1:] contienen las fronteras de los agujeros.

        Por ejemplo:

            fronteras = [
                [F0, F1, F2, F3],   # exterior
                [F4, F5, F6, F7],   # agujero 1
            ]

        Cada frontera contiene los índices de xnode correspondientes
        a un segmento del polígono.

    mostrar_numeros : bool, opcional
        Si True, muestra el índice de cada nodo.

    mostrar_nodos : bool, opcional
        Si True, dibuja los nodos.

    mostrar_elementos : bool, opcional
        Si True, muestra el índice de cada elemento.

    mostrar_fronteras : bool, opcional
        Si True y fronteras no es None, colorea cada frontera
        con un color diferente y muestra su número.

    Retorna
    -------
    fig : matplotlib.figure.Figure
        Figura creada.

    ax : matplotlib.axes.Axes
        Ejes utilizados.
    """

    # =========================================================
    # Conversión y validación
    # =========================================================

    xnode = np.asarray(xnode, dtype=float)
    icone = np.asarray(icone, dtype=int)

    if xnode.ndim != 2 or xnode.shape[1] != 2:
        raise ValueError(
            "xnode debe tener forma (nnodes, 2)."
        )

    if icone.ndim != 2 or icone.shape[1] != 4:
        raise ValueError(
            "icone debe tener forma (nelem, 4)."
        )
    
    # ---------------------------------------------------------
    # Polígonos
    # ---------------------------------------------------------

    if poligonos is not None:

        if len(poligonos) == 0:
            raise ValueError(
                "poligonos no puede estar vacío."
            )

        for i, poligono in enumerate(poligonos):

            poligono = np.asarray(
                poligono,
                dtype=float
            )

            if poligono.ndim != 2 or poligono.shape[1] != 2:
                raise ValueError(
                    f"El polígono {i} debe tener forma (n, 2)."
                )

            if len(poligono) < 3:
                raise ValueError(
                    f"El polígono {i} debe tener al menos 3 vértices."
                )

            # Guardamos la conversión para trabajar siempre
            # con arrays de numpy.
            poligonos[i] = poligono
    
    # ---------------------------------------------------------
    # Fronteras
    # ---------------------------------------------------------

    if fronteras is not None:

        if poligonos is not None:

            if len(fronteras) != len(poligonos):
                raise ValueError(
                    "La cantidad de grupos de fronteras debe "
                    "coincidir con la cantidad de polígonos."
                )

            for i, (poligono, fronteras_poligono) in enumerate(
                zip(poligonos, fronteras)
            ):

                if len(fronteras_poligono) != len(poligono):
                    raise ValueError(
                        f"El polígono {i} tiene "
                        f"{len(poligono)} lados, pero se encontraron "
                        f"{len(fronteras_poligono)} fronteras."
                    )

    # =========================================================
    # Límites de la malla
    # =========================================================

    x_min = np.min(xnode[:, 0])
    x_max = np.max(xnode[:, 0])

    y_min = np.min(xnode[:, 1])
    y_max = np.max(xnode[:, 1])

    dx = x_max - x_min
    dy = y_max - y_min

    # =========================================================
    # Figura
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    # =========================================================
    # Dibujar elementos
    # =========================================================

    for e, elem in enumerate(icone):

        # Cerrar el cuadrilátero
        nodos = np.append(elem, elem[0])

        x = xnode[nodos, 0]
        y = xnode[nodos, 1]

        ax.plot(
            x,
            y,
            color="0.35",
            linewidth=1,
            zorder=1
        )

        # -----------------------------------------------------
        # Número del elemento
        # -----------------------------------------------------

        if mostrar_elementos:

            xc = np.mean(
                xnode[elem, 0]
            )

            yc = np.mean(
                xnode[elem, 1]
            )

            ax.annotate(
                str(e),
                xy=(xc, yc),
                ha="center",
                va="center",
                fontsize=8,
                color="tab:blue",
                zorder=3
            )

    # =========================================================
    # Dibujar nodos
    # =========================================================

    if mostrar_nodos:

        ax.scatter(
            xnode[:, 0],
            xnode[:, 1],
            color="red",
            marker="+",
            s=50,
            linewidths=1,
            zorder=4
        )

    # =========================================================
    # Numeración de nodos
    # =========================================================

    if mostrar_numeros:

        for i, (x, y) in enumerate(xnode):

            ax.annotate(
                str(i),
                xy=(x, y),
                xytext=(8, 8),
                textcoords="offset points",
                va="center",
                ha="center",
                fontsize=8,
                zorder=5
            )

    # =========================================================
    # Relleno del dominio
    # =========================================================

    if poligonos is not None:

        # -----------------------------------------------------
        # Rellenar dominio exterior
        # -----------------------------------------------------

        exterior = poligonos[0]

        ax.fill(
            exterior[:, 0],
            exterior[:, 1],
            color="tab:blue",
            alpha=0.06,
            zorder=0
        )

        # -----------------------------------------------------
        # Vaciar visualmente los agujeros
        # -----------------------------------------------------

        for hueco in poligonos[1:]:

            ax.fill(
                hueco[:, 0],
                hueco[:, 1],
                color="white",
                zorder=0
            )

    # =========================================================
    # Fronteras coloreadas
    # =========================================================

    if (
        mostrar_fronteras
        and fronteras is not None
    ):

        # -----------------------------------------------------
        # Colores
        # -----------------------------------------------------

        # tab10 proporciona diez colores claramente
        # diferenciables.
        cmap = plt.get_cmap("tab10")
        
        # Número global de frontera
        numero_frontera = 0

        # -----------------------------------------------------
        # Recorrer polígonos
        # -----------------------------------------------------

        for fronteras_poligono in fronteras:

            # -------------------------------------------------
            # Recorrer fronteras de cada polígono
            # -------------------------------------------------

            for nodos in fronteras_poligono:

                nodos = np.asarray(
                    nodos,
                    dtype=int
                )

                if len(nodos) == 0:
                    continue

                color = cmap(numero_frontera % 10)

                # -------------------------------------------------
                # Coordenadas de los nodos de la frontera
                # -------------------------------------------------

                x = xnode[nodos, 0]
                y = xnode[nodos, 1]

                # -------------------------------------------------
                # Dibujar frontera
                # -------------------------------------------------

                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=4,
                    solid_capstyle="round",
                    zorder=6
                )

                # -------------------------------------------------
                # Determinar posición de la etiqueta
                # -------------------------------------------------

                if len(nodos) >= 2:

                    puntos = xnode[nodos]

                    diferencias = np.diff(
                        puntos,
                        axis=0
                    )

                    longitudes = np.sqrt(
                        np.sum(
                            diferencias**2,
                            axis=1
                        )
                    )

                    longitud_total = np.sum(
                        longitudes
                    )

                    if longitud_total > 0:

                        distancia_objetivo = (
                            longitud_total / 2
                        )

                        acumulada = 0.0

                        for k, longitud in enumerate(
                            longitudes
                        ):

                            if (
                                acumulada + longitud
                                >= distancia_objetivo
                            ):

                                fraccion = (
                                    distancia_objetivo
                                    - acumulada
                                ) / longitud

                                punto_medio = (
                                    puntos[k]
                                    + fraccion
                                    * (
                                        puntos[k + 1]
                                        - puntos[k]
                                    )
                                )

                                break

                            acumulada += longitud

                        else:

                            punto_medio = puntos[-1]

                    else:

                        punto_medio = puntos[0]

                else:

                    punto_medio = xnode[nodos[0]]

                # -------------------------------------------------
                # Etiqueta F0, F1, ...
                # -------------------------------------------------

                ax.annotate(
                    f"F{numero_frontera}",
                    xy=punto_medio,
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=color,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor=color,
                        linewidth=1,
                        alpha=0.9
                    ),
                    zorder=8
                )

                numero_frontera += 1

    # =========================================================
    # Contorno original
    # =========================================================

    if poligonos is not None:

        for poligono in poligonos:

            poligono_cerrado = np.vstack([
                poligono,
                poligono[0]
            ])

            ax.plot(
                poligono_cerrado[:, 0],
                poligono_cerrado[:, 1],
                color="black",
                linewidth=2,
                zorder=7
            )

    # =========================================================
    # Configuración de ejes
    # =========================================================

    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # ---------------------------------------------------------
    # Márgenes
    # ---------------------------------------------------------

    margen_x = (
        0.05 * dx
        if dx > 0
        else 1
    )

    margen_y = (
        0.05 * dy
        if dy > 0
        else 1
    )

    ax.set_xlim(
        x_min - margen_x,
        x_max + margen_x
    )

    ax.set_ylim(
        y_min - margen_y,
        y_max + margen_y
    )

    ax.grid(False)

    # =========================================================
    # Título
    # =========================================================

    ax.set_title(
        f"Malla de {len(icone)} elementos y "
        f"{len(xnode)} nodos"
    )

    # =========================================================
    # Mostrar
    # =========================================================

    plt.tight_layout()
    plt.show()

    return fig, ax