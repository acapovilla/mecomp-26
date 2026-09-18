import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.tri as mtri

from enum import IntEnum

class GraphType(IntEnum):
    TEMPERATURE = 0
    HEAT_FLUX = 1
    HEAT_FLUX_COMPONENTS = 2


def plot_temperature(xnode, icone, PHI, *, mesh=True, shading="gouraud", margin=0.05, cmap="coolwarm"):
    """
    Plot the stationary temperature field.

    Parameters
    ----------
    xnode : ndarray, shape (N, 2)
        Node coordinates. Each row contains [x, y].

    icone : ndarray, shape (M, 4)
        Connectivity matrix. Each row contains the four node
        indices of a quadrilateral element.

    PHI : ndarray, shape (N,)
        Nodal temperature solution.

    mesh : bool, optional
        If True, display the mesh over the temperature field.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.

    ax : matplotlib.axes.Axes
        Axes object.
    """

    xnode = np.asarray(xnode)
    icone = np.asarray(icone)
    PHI = np.asarray(PHI).reshape(-1)

    # Node coordinates
    X = xnode[:, 0]
    Y = xnode[:, 1]

    # Convert quadrilateral elements into triangles
    triangles = np.empty((2 * len(icone), 3), dtype=int)

    triangles[:len(icone), :] = icone[:, [0, 1, 2]]
    triangles[len(icone):, :] = icone[:, [0, 2, 3]]

    # Create triangulation
    triangulation = mtri.Triangulation(X, Y, triangles)

    # Figure
    fig, ax = plt.subplots(figsize=(12,7))

    # Temperature field
    contour = ax.tripcolor(triangulation, PHI, cmap=cmap, shading=shading)

    # Mesh
    if mesh:
        ax.triplot(triangulation, color="k", linewidth=0.5)

    # Colorbar
    fig.colorbar(contour, ax=ax, label="Temperature")
    
    # Plot limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    dx = xmax - xmin
    dy = ymax - ymin

    ax.set_xlim(xmin - margin * dx, xmax + margin * dx)
    ax.set_ylim(ymin - margin * dy, ymax + margin * dy)

    # Axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    ax.set_title("Temperature distribution")

    plt.show()

    return fig, ax

def plot_temperature_transient(
    xnode,
    icone,
    PHI,
    delta_t,
    *,
    mesh=True,
    shading="gouraud",
    cmap="coolwarm",
    margin=0.05,
    fps=25,
    frames=None,
    save_path=None,
):
    xnode = np.asarray(xnode)
    icone = np.asarray(icone)
    PHI = np.asarray(PHI)

    if PHI.ndim != 2:
        raise ValueError(
            "PHI must contain one temperature vector per time step."
        )

    if PHI.shape[1] != len(xnode):
        raise ValueError(
            "Each PHI vector must contain one value per node."
        )
    
    # Tiempo asociado a cada solución
    time = np.arange(len(PHI)) * delta_t

    X = xnode[:, 0]
    Y = xnode[:, 1]

    n_elements = len(icone)

    triangles = np.empty((2 * n_elements, 3), dtype=int)
    triangles[:n_elements] = icone[:, [0, 1, 2]]
    triangles[n_elements:] = icone[:, [0, 2, 3]]

    triangulation = mtri.Triangulation(X, Y, triangles)

    # Límites espaciales
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    dx = xmax - xmin
    dy = ymax - ymin

    xlim = (xmin - margin * dx, xmax + margin * dx)

    ylim = (ymin - margin * dy, ymax + margin * dy)

    # Escala de colores fija para toda la animación
    vmin = np.min(PHI)
    vmax = np.max(PHI)

    fig, ax = plt.subplots(figsize=(12,7))

    contour = ax.tripcolor(triangulation, PHI[0, :], cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)

    if mesh:
        ax.triplot(triangulation, color="k", linewidth=0.5)

    fig.colorbar(contour, ax=ax, label="Temperature", location='left')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    def animate(i):
        contour.set_array(PHI[i])
        ax.set_title(f"Temperature - t = {time[i]:.3f} s")

        return contour,
    
    if frames is None:
        frame_indices = np.arange(len(PHI))

    else:
        if not isinstance(frames, (int, np.integer)):
            raise TypeError(
                "frames must be an integer or None."
            )

        if frames < 2:
            raise ValueError(
                "frames must be at least 2."
            )

        if frames > len(PHI):
            raise ValueError(
                "frames cannot exceed the number of time steps."
            )

        frame_indices = np.linspace(
            0,
            len(PHI) - 1,
            frames,
            dtype=int,
        )

    anim = FuncAnimation(
        fig,
        animate,
        frames=frame_indices,
        interval=1000 / fps,
        blit=False,
    )

    if save_path is not None:
        anim.save(
            save_path,
            writer=PillowWriter(fps=fps, bitrate=-1),
        )

    return fig, anim

def plot_temperature_3d(
    xnode,
    icone,
    PHI,
    *,
    mesh=True,
    cmap="coolwarm",
    margin=0.05,
    vmin=None,
    vmax=None,
    elev=30,
    azim=-60,
):
    xnode = np.asarray(xnode)
    icone = np.asarray(icone)
    PHI = np.asarray(PHI).reshape(-1)

    if PHI.shape[0] != len(xnode):
        raise ValueError(
            "PHI must contain one value per node."
        )

    X = xnode[:, 0]
    Y = xnode[:, 1]

    if vmin is None:
        vmin = np.min(PHI)

    if vmax is None:
        vmax = np.max(PHI)

    # Límites espaciales
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    dx = xmax - xmin
    dy = ymax - ymin

    xlim = (
        xmin - margin * dx,
        xmax + margin * dx,
    )

    ylim = (
        ymin - margin * dy,
        ymax + margin * dy,
    )

    fig, ax = plt.subplots(
        figsize=(12,7),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    surface = ax.plot_trisurf(
        X,
        Y,
        PHI,
        triangles=np.vstack(
            (
                icone[:, [0, 1, 2]],
                icone[:, [0, 2, 3]],
            )
        ),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0 if not mesh else 0.5,
        edgecolor="k" if mesh else "none",
    )

    fig.colorbar(
        surface,
        ax=ax,
        # label="Temperature",
        shrink=0.7,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Temperature")
    ax.set_aspect('equalxy')

    ax.set_title("Temperature distribution")

    ax.view_init(
        elev=elev,
        azim=azim,
    )

    return fig, ax

def plot_temperature_3d_transient(
    xnode,
    icone,
    PHI,
    delta_t,
    *,
    mesh=True,
    cmap="coolwarm",
    margin=0.05,
    vmin=None,
    vmax=None,
    azim=-60,
    fps=25,
    frames=None,
    save_path=None,
):
    xnode = np.asarray(xnode)
    icone = np.asarray(icone)
    PHI = np.asarray(PHI)

    if PHI.ndim != 2:
        raise ValueError(
            "PHI must contain one temperature vector per time step."
        )

    if PHI.shape[1] != len(xnode):
        raise ValueError(
            "Each PHI vector must contain one value per node."
        )
    
    # Tiempo asociado a cada solución
    time = np.arange(len(PHI)) * delta_t

    X = xnode[:, 0]
    Y = xnode[:, 1]
    
    n_elements = len(icone)

    triangles = np.empty((2 * n_elements, 3), dtype=int)
    triangles[:n_elements] = icone[:, [0, 1, 2]]
    triangles[n_elements:] = icone[:, [0, 2, 3]]

    # Límites espaciales
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    dx = xmax - xmin
    dy = ymax - ymin

    xlim = (
        xmin - margin * dx,
        xmax + margin * dx,
    )

    ylim = (
        ymin - margin * dy,
        ymax + margin * dy,
    )
    
    # Escala de colores fija para toda la animación
    if vmin is None:
        vmin = np.min(PHI)

    if vmax is None:
        vmax = np.max(PHI)

    fig, ax = plt.subplots(
        figsize=(12,7),
        subplot_kw={"projection": "3d"},
        # constrained_layout=True,
    )

    surface = ax.plot_trisurf(
        X,
        Y,
        PHI[0],
        triangles=np.vstack(
            (
                icone[:, [0, 1, 2]],
                icone[:, [0, 2, 3]],
            )
        ),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0 if not mesh else 0.5,
        edgecolor="k" if mesh else "none",
    )

    _ = fig.colorbar(
        surface,
        ax=ax,
        # pad=0.1,
        shrink=0.7,
        # label="Temperature",
    )
    
    # cbar.ax.set_title(
    #     "Temperature",
    #     fontsize=10,
    #     pad=5,
    #     loc="center",
    # )
    
    # Ajustar la distribución del área de dibujo
    fig.subplots_adjust(
        left=0.0,
        right=0.85,
        bottom=0.05,
        top=0.95,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(vmin, vmax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Temperature")
    ax.set_aspect('equalxy')

    ax.set_title("Temperature distribution")
    time_text = ax.text2D(
        0.05,
        0.95,
        f"t = {time[0]:.2f} s",
        transform=ax.transAxes,
        fontsize=12,
    )

    ax.view_init(
        elev=30,
        azim=azim,
    )

    def animate(i):
        nonlocal surface

        # Eliminar la superficie anterior
        surface.remove()

        # Crear la nueva superficie
        surface = ax.plot_trisurf(
            X,
            Y,
            PHI[i],
            triangles=triangles,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0 if not mesh else 0.5,
            edgecolor="k" if mesh else "none",
        )

        # ax.set_title(f"Temperature - t = {time[i]:.3f} s")
        time_text.set_text(f"t = {time[i]:.2f} s")

        return surface, time_text
    
    if frames is None:
        frame_indices = np.arange(len(PHI))

    else:
        if not isinstance(frames, (int, np.integer)):
            raise TypeError(
                "frames must be an integer or None."
            )

        if frames < 2:
            raise ValueError(
                "frames must be at least 2."
            )

        if frames > len(PHI):
            raise ValueError(
                "frames cannot exceed the number of time steps."
            )

        frame_indices = np.linspace(
            0,
            len(PHI) - 1,
            frames,
            dtype=int,
        )

    anim = FuncAnimation(
        fig,
        animate,
        frames=frame_indices,
        interval=1000 / fps,
        blit=False,
    )

    if save_path is not None:
        anim.save(
            save_path,
            writer=PillowWriter(fps=fps, bitrate=-1),
        )

    return fig, anim

def plot_heat_flux(xnode, icone, PHI, Q, *, mesh=True, shading="gouraud", margin=0.05, scale=None, linewidth=0.004, cmap="coolwarm"):
    """
    Plot the stationary heat-flux field.

    Parameters
    ----------
    xnode : ndarray, shape (N, 2)
        Node coordinates. Each row contains [x, y].

    icone : ndarray, shape (M, 4)
        Connectivity matrix. Each row contains the four node
        indices of a quadrilateral element.

    PHI : ndarray, shape (N,)
        Nodal temperature solution. Used as the background
        scalar field.

    Q : ndarray, shape (N, 2)
        Nodal heat flux. Columns are [qx, qy].

    mesh : bool, optional
        If True, display the mesh over the background field.

    shading : {"gouraud", "flat"}, optional
        Interpolation used for the background temperature field.

    margin : float, optional
        Relative margin around the mesh. A value of 0 means
        that the axes are exactly bounded by the mesh.

    scale : float or None, optional
        Scaling factor used by quiver. If None, matplotlib
        determines the scale automatically.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.

    ax : matplotlib.axes.Axes
        Axes object.
    """

    xnode = np.asarray(xnode)
    icone = np.asarray(icone)
    PHI = np.asarray(PHI).reshape(-1)
    Q = np.asarray(Q)

    # Node coordinates
    X = xnode[:, 0]
    Y = xnode[:, 1]

    # Check dimensions
    if len(PHI) != len(xnode):
        raise ValueError("PHI must contain one value per node.")

    if Q.shape != (len(xnode), 2):
        raise ValueError("Q must have shape (number_of_nodes, 2).")

    # Convert quadrilateral elements into triangles
    n_elements = len(icone)

    triangles = np.empty((2 * n_elements, 3), dtype=int)

    triangles[:n_elements] = icone[:, [0, 1, 2]]
    triangles[n_elements:] = icone[:, [0, 2, 3]]

    # Create triangulation
    triangulation = mtri.Triangulation(X, Y, triangles)

    # Create figure
    fig, ax = plt.subplots(figsize=(12,7))

    # Temperature background
    contour = ax.tripcolor(triangulation, PHI, cmap=cmap, shading=shading)

    # Heat flux components
    Qx = Q[:, 0]
    Qy = Q[:, 1]

    # Heat-flux vectors
    ax.quiver(X, Y, Qx, Qy, scale=scale, width=linewidth)

    # Mesh
    if mesh:
        ax.triplot(triangulation, color="black", linewidth=0.5)

    # Colorbar for temperature
    fig.colorbar(contour, ax=ax, label="Temperature")

    # Plot limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    dx = xmax - xmin
    dy = ymax - ymin

    ax.set_xlim(xmin - margin * dx, xmax + margin * dx)
    ax.set_ylim(ymin - margin * dy, ymax + margin * dy)

    # Axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    ax.set_title("Heat flux and temperature distribution")

    plt.show()

    return fig, ax

def plot_heat_flux_transient(
    xnode,
    icone,
    PHI,
    Q,
    delta_t,
    *,
    mesh=True,
    shading="gouraud",
    cmap="coolwarm",
    margin=0.05,
    fps=25,
    frames=None,
    scale=None,
    width=0.005,
    save_path=None,
):
    xnode = np.asarray(xnode)
    icone = np.asarray(icone)
    PHI = np.asarray(PHI)
    Q = np.asarray(Q)

    if PHI.ndim != 2:
        raise ValueError(
            "PHI must contain one temperature vector per time step."
        )

    if PHI.shape[1] != len(xnode):
        raise ValueError(
            "Each PHI vector must contain one value per node."
        )

    if Q.ndim != 3 or Q.shape[2] != 2:
        raise ValueError(
            "Q must contain one (qx, qy) vector per node and time step."
        )

    if Q.shape[0] != PHI.shape[0]:
        raise ValueError(
            "PHI and Q must contain the same number of time steps."
        )

    if Q.shape[1] != len(xnode):
        raise ValueError(
            "Each Q vector must contain one (qx, qy) value per node."
        )

    # Tiempo asociado a cada solución
    time = np.arange(len(PHI)) * delta_t

    X = xnode[:, 0]
    Y = xnode[:, 1]

    n_elements = len(icone)

    triangles = np.empty((2 * n_elements, 3), dtype=int)
    triangles[:n_elements] = icone[:, [0, 1, 2]]
    triangles[n_elements:] = icone[:, [0, 2, 3]]

    triangulation = mtri.Triangulation(X, Y, triangles)

    # Límites espaciales
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    dx = xmax - xmin
    dy = ymax - ymin

    xlim = (xmin - margin * dx, xmax + margin * dx)
    ylim = (ymin - margin * dy, ymax + margin * dy)

    # Escala de colores fija para toda la animación
    vmin = np.min(PHI)
    vmax = np.max(PHI)

    fig, ax = plt.subplots(figsize=(12,7))

    # Temperatura como fondo
    contour = ax.tripcolor(
        triangulation,
        PHI[0],
        cmap=cmap,
        shading=shading,
        vmin=vmin,
        vmax=vmax,
    )

    # Malla
    if mesh:
        ax.triplot(
            triangulation,
            color="k",
            linewidth=0.5,
        )

    fig.colorbar(
        contour,
        ax=ax,
        label="Temperature",
    )

    # Flujo térmico inicial
    Qx = Q[0, :, 0]
    Qy = Q[0, :, 1]

    quiver = ax.quiver(
        X,
        Y,
        Qx,
        Qy,
        scale=scale,
        width=width,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    def animate(i):
        # Actualizar temperatura
        contour.set_array(PHI[i])

        # Actualizar flujo
        quiver.set_UVC(
            Q[i, :, 0],
            Q[i, :, 1],
        )

        ax.set_title(
            f"Heat flux - t = {time[i]:.3f}"
        )

        return contour, quiver
    
    if frames is None:
        frame_indices = np.arange(len(PHI))

    else:
        if not isinstance(frames, (int, np.integer)):
            raise TypeError(
                "frames must be an integer or None."
            )

        if frames < 2:
            raise ValueError(
                "frames must be at least 2."
            )

        if frames > len(PHI):
            raise ValueError(
                "frames cannot exceed the number of time steps."
            )

        frame_indices = np.linspace(
            0,
            len(PHI) - 1,
            frames,
            dtype=int,
        )

    anim = FuncAnimation(
        fig,
        animate,
        frames=frame_indices,
        interval=1000 / fps,
        blit=False,
    )

    if save_path is not None:
        anim.save(
            save_path,
            writer=PillowWriter(fps=fps),
        )

    return fig, anim

def plot_heat_flux_3d(
    xnode,
    icone,
    PHI,
    Q,
    *,
    mesh=True,
    cmap="coolwarm",
    margin=0.05,
    vmin=None,
    vmax=None,
    azim=-60,
    arrow_scale=0.1,
):
    xnode = np.asarray(xnode)
    icone = np.asarray(icone)
    PHI = np.asarray(PHI).reshape(-1)
    Q = np.asarray(Q)

    if PHI.shape[0] != len(xnode):
        raise ValueError(
            "PHI must contain one value per node."
        )

    if Q.shape != (len(xnode), 2):
        raise ValueError(
            "Q must have shape (number_of_nodes, 2)."
        )

    X = xnode[:, 0]
    Y = xnode[:, 1]

    if vmin is None:
        vmin = np.min(PHI)

    if vmax is None:
        vmax = np.max(PHI)

    # Límites espaciales
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    dx = xmax - xmin
    dy = ymax - ymin

    xlim = (
        xmin - margin * dx,
        xmax + margin * dx,
    )

    ylim = (
        ymin - margin * dy,
        ymax + margin * dy,
    )

    # Triangulación
    n_elements = len(icone)

    triangles = np.empty(
        (2 * n_elements, 3),
        dtype=int,
    )

    triangles[:n_elements] = icone[:, [0, 1, 2]]
    triangles[n_elements:] = icone[:, [0, 2, 3]]

    fig, ax = plt.subplots(
        figsize=(12,7),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    # Superficie de temperatura
    surface = ax.plot_trisurf(
        X,
        Y,
        PHI,
        triangles=triangles,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0 if not mesh else 0.5,
        edgecolor="k" if mesh else "none",
    )

    fig.colorbar(
        surface,
        ax=ax,
        label="Temperature",
        shrink=0.7,
    )

    # Flujo térmico
    Qx = Q[:, 0]
    Qy = Q[:, 1]
    
    Qmag = np.linalg.norm(Q, axis=1)
    qmax = np.max(Qmag)
    domain_size = max(dx, dy)
    
    length = arrow_scale * domain_size / qmax

    quiver = ax.quiver(
        X,
        Y,
        PHI,
        Qx,
        Qy,
        np.zeros_like(Qx),
        color='k',
        length=length,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Temperature")
    ax.set_aspect('equalxy')

    ax.set_title("Heat flux and temperature distribution")

    ax.view_init(
        elev=30,
        azim=azim,
    )

    return fig, ax

def plot_heat_flux_components(xnode, icone, Q, *, mesh=True, shading="gouraud", margin=0.05, cmap="coolwarm"):
    """
    Plot the stationary heat-flux components and magnitude.

    Parameters
    ----------
    xnode : ndarray, shape (N, 2)
        Node coordinates. Each row contains [x, y].

    icone : ndarray, shape (M, 4)
        Connectivity matrix. Each row contains the four node
        indices of a quadrilateral element.

    Q : ndarray, shape (N, 2)
        Nodal heat flux. Columns are [qx, qy].

    mesh : bool, optional
        If True, display the mesh over each field.

    shading : {"gouraud", "flat"}, optional
        Interpolation used for the scalar fields.

    margin : float, optional
        Relative margin around the mesh. A value of 0 means
        that the axes are exactly bounded by the mesh.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.

    axes : ndarray
        Array containing the three Axes objects.
    """

    xnode = np.asarray(xnode)
    icone = np.asarray(icone)
    Q = np.asarray(Q)

    # Node coordinates
    X = xnode[:, 0]
    Y = xnode[:, 1]

    # Check dimensions
    if Q.shape != (len(xnode), 2):
        raise ValueError("Q must have shape (number_of_nodes, 2).")

    # Convert quadrilateral elements into triangles
    n_elements = len(icone)

    triangles = np.empty((2 * n_elements, 3), dtype=int)

    triangles[:n_elements] = icone[:, [0, 1, 2]]
    triangles[n_elements:] = icone[:, [0, 2, 3]]

    # Create triangulation
    triangulation = mtri.Triangulation(X, Y, triangles)

    # Flux components
    Qx = Q[:, 0]
    Qy = Q[:, 1]

    # Flux magnitude
    Qmag = np.sqrt(Qx**2 + Qy**2)

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    fields = [
        (Qx, r"$q_x$"),
        (Qy, r"$q_y$"),
        (Qmag, r"$|\mathbf{q}|$")
    ]

    # Plot limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    dx = xmax - xmin
    dy = ymax - ymin

    xlim = (xmin - margin * dx, xmax + margin * dx)
    ylim = (ymin - margin * dy, ymax + margin * dy)

    # Plot each field
    for ax, (field, title) in zip(axes, fields):
        contour = ax.tripcolor(triangulation, field, cmap=cmap, shading=shading)

        if mesh:
            ax.triplot(triangulation, color="k", linewidth=0.5)

        fig.colorbar(contour, ax=ax)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

        ax.set_aspect("equal")

    fig.suptitle("Heat flux")

    plt.show()

    return fig, axes

def fdm2d_graph_mesh(
    xnode,
    icone,
    PHI,
    Q=None,
    delta_t=None,
    *,
    graph=GraphType.TEMPERATURE,
    dimension=2,
    transient=False,
    mesh=True,
    shading="gouraud",
    margin=0.05,
    **kwargs
):
    """
    Frontend for 2D finite-difference results visualization.

    Parameters
    ----------
    xnode : ndarray, shape (N, 2)
        Node coordinates [x, y].

    icone : ndarray, shape (M, 4)
        Element connectivity.

    PHI : ndarray, shape (N,)
        Nodal temperature solution.

    Q : ndarray, shape (N, 2), optional
        Nodal heat flux [qx, qy].

    graph : GraphType, optional
        Type of result to display.

    mesh : bool, optional
        Display the mesh.

    shading : {"gouraud", "flat"}, optional
        Interpolation used for scalar fields.

    margin : float, optional
        Relative margin around the mesh.

    **kwargs
        Additional plotting options passed to the selected
        visualization function.
    """
    
    if dimension not in (2, 3):
        raise ValueError(
            "dimension must be 2 or 3."
        )

    if transient and delta_t is None:
        raise ValueError(
            "delta_t is required for transient solutions."
        )

    if graph == GraphType.TEMPERATURE:
        if transient:
            if dimension == 2:
                return plot_temperature_transient(
                    xnode,
                    icone,
                    PHI,
                    delta_t,
                    mesh=mesh,
                    shading=shading,
                    margin=margin,
                    **kwargs,
                )
            else:  # dimension == 3
                return plot_temperature_3d_transient(
                    xnode,
                    icone,
                    PHI,
                    delta_t,
                    mesh=mesh,
                    margin=margin,
                    **kwargs,
                )
        else:
            if dimension == 2:
                return plot_temperature(
                    xnode,
                    icone,
                    PHI,
                    mesh=mesh,
                    shading=shading,
                    margin=margin,
                    **kwargs,
                )
            else:  # dimension == 3
                return plot_temperature_3d(
                    xnode,
                    icone,
                    PHI,
                    mesh=mesh,
                    margin=margin,
                    **kwargs,
                )
            

    elif graph == GraphType.HEAT_FLUX:

        if Q is None:
            raise ValueError(
                "Q is required to plot the heat flux."
            )
        
        if transient:
            raise NotImplementedError(
                "Transient heat flux plotting "
                "is not implemented yet."
            )
            
        if dimension == 2:
            return plot_heat_flux(
                xnode,
                icone,
                PHI,
                Q,
                mesh=mesh,
                shading=shading,
                margin=margin,
                **kwargs,
            )
        else:
            return plot_heat_flux_3d(
                xnode,
                icone,
                PHI,
                Q,
                mesh=mesh,
                margin=margin,
                **kwargs,
            )

    elif graph == GraphType.HEAT_FLUX_COMPONENTS:

        if Q is None:
            raise ValueError(
                "Q is required to plot the heat flux components."
            )

        return plot_heat_flux_components(
            xnode,
            icone,
            Q,
            mesh=mesh,
            shading=shading,
            margin=margin,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown graph type: {graph}"
        )
