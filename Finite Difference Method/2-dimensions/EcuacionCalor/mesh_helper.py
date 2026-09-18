import numpy as np

from .mesh_maker import (
    separar_poligonos,
    construir_eje,
    recortar_grilla,
    obtener_nodos_frontera,
    dibujar_malla,
)

def primitiva_rect(x_pos=0.0, y_pos=0.0, x_scale=1.0, y_scale=1.0, x_subdiv=0, y_subdiv=0, plot=False, tol=1e-12):
    vertices = np.array([
        [    0.0,     0.0],
        [    1.0,     0.0],
        [    1.0,     1.0],
        [    0.0,     1.0],
        [    0.0,     0.0],
    ])
    
    vertices[:, 0] = vertices[:, 0] * x_scale + x_pos
    vertices[:, 1] = vertices[:, 1] * y_scale + y_pos
    
    poligonos = separar_poligonos(vertices=vertices, tol=tol)
    
    x_geom = np.array([0.0, 1.0/3.0, 2.0/3.0, 1.0])
    y_geom = x_geom
    
    x_geom = x_geom * x_scale + x_pos
    y_geom = y_geom * y_scale + y_pos
    
    xx = construir_eje(coordenadas=x_geom, niveles=x_subdiv)
    yy = construir_eje(coordenadas=y_geom, niveles=y_subdiv)

    xnode, icone, P = recortar_grilla(poligonos=poligonos, xx=xx, yy=yy)
    
    fronteras = obtener_nodos_frontera(
        poligonos=poligonos,
        xnode=xnode,
        tol=tol
    )
    
    if plot:
        dibujar_malla(xnode, icone, poligonos, fronteras, mostrar_numeros=True, mostrar_elementos=True)
    
    return xnode, icone, vertices, fronteras[0]

def primitiva_L(x_pos=0.0, y_pos=0.0, x_scale=1.0, y_scale=1.0, x_subdiv=0, y_subdiv=0, plot=False, tol=1e-12):
    vertices = np.array([
        [    0.0,     0.0],
        [    1.0,     0.0],
        [    1.0, 1.0/3.0],
        [1.0/3.0, 1.0/3.0],
        [1.0/3.0,     1.0],
        [    0.0,     1.0],
        [    0.0,     0.0], 
    ])
    
    vertices[:, 0] = vertices[:, 0] * x_scale + x_pos
    vertices[:, 1] = vertices[:, 1] * y_scale + y_pos
    
    poligonos = separar_poligonos(vertices=vertices, tol=tol)
    
    x_geom = np.array([0.0, 1.0/3.0, 2.0/3.0, 1.0])
    y_geom = x_geom
    
    x_geom = x_geom * x_scale + x_pos
    y_geom = y_geom * y_scale + y_pos
    
    xx = construir_eje(coordenadas=x_geom, niveles=x_subdiv)
    yy = construir_eje(coordenadas=y_geom, niveles=y_subdiv)

    xnode, icone, P = recortar_grilla(poligonos=poligonos, xx=xx, yy=yy)
    
    fronteras = obtener_nodos_frontera(
        poligonos=poligonos,
        xnode=xnode,
        tol=tol
    )
    
    if plot:
        dibujar_malla(xnode, icone, poligonos, fronteras, mostrar_numeros=True, mostrar_elementos=True)
    
    return xnode, icone, vertices, fronteras[0]

def primitiva_O(x_pos=0.0, y_pos=0.0, x_scale=1.0, y_scale=1.0, x_subdiv=0, y_subdiv=0, plot=False, tol=1e-12):
    vertices = np.array([
        # Exterior
        [    0.0,     0.0],
        [    1.0,     0.0],
        [    1.0,     1.0],
        [    0.0,     1.0],
        [    0.0,     0.0],
        # Hueco interior
        [1.0/3.0, 1.0/3.0],
        [2.0/3.0, 1.0/3.0],
        [2.0/3.0, 2.0/3.0],
        [1.0/3.0, 2.0/3.0],
        [1.0/3.0, 1.0/3.0],
    ])
    
    vertices[:, 0] = vertices[:, 0] * x_scale + x_pos
    vertices[:, 1] = vertices[:, 1] * y_scale + y_pos
    
    poligonos = separar_poligonos(vertices=vertices, tol=tol)
    
    x_geom = np.array([0.0, 1.0/3.0, 2.0/3.0, 1.0])
    y_geom = x_geom
    
    x_geom = x_geom * x_scale + x_pos
    y_geom = y_geom * y_scale + y_pos
    
    xx = construir_eje(coordenadas=x_geom, niveles=x_subdiv)
    yy = construir_eje(coordenadas=y_geom, niveles=y_subdiv)

    xnode, icone, P = recortar_grilla(poligonos=poligonos, xx=xx, yy=yy)
    
    fronteras = obtener_nodos_frontera(
        poligonos=poligonos,
        xnode=xnode,
        tol=tol
    )
    
    if plot:
        dibujar_malla(xnode, icone, poligonos, fronteras, mostrar_numeros=True, mostrar_elementos=True)
    
    return xnode, icone, vertices, fronteras
