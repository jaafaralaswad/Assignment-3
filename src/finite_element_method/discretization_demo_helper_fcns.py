from finite_element_method import discretization as di
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_gauss_pts(fname, ele_type, num_pts):
    """
    Visualizes Gauss quadrature points and element nodes in natural coordinates.
    
    Parameters
    ----------
    fname : str
        The filename to save the plot.
    ele_type : str
        The type of finite element (e.g., "D3_nn8_hex").
    num_pts : int
        The number of Gauss integration points.
    
    Saves
    -----
    A figure displaying the element's reference shape with labeled nodes, mid-edge nodes (if applicable),
    and Gauss points.
    """
    # Get Gauss points
    gauss_pts, _ = gauss_pts_and_weights(ele_type, num_pts)
    gauss_pts = gauss_pts.T

    # Define reference element nodes in natural coordinates
    if ele_type == "D3_nn8_hex":
        nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]])  # Brick
        edges = [[0, 1], [0, 3], [0, 4],[1, 2], [1, 5], [2, 3],[2, 6], [3, 7], [4, 5],[4, 7], [5, 6], [6, 7]]
    else:
        raise ValueError(f"Unsupported element type: {ele_type}")
    
    # 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot element edges
    for edge in edges:
        ax.plot(nodes[edge, 0], nodes[edge, 1], nodes[edge, 2], 'k-', lw=2)
    
    # Plot and label element nodes
    for i, (x, y, z) in enumerate(nodes):
        ax.scatter(x, y, z, color='blue', s=100, edgecolors='k', zorder=3)
        ax.text(x, y, z, f'N{i+1}', fontsize=12, ha='right', va='bottom', color='blue', bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot and label Gauss points
    for i in range(gauss_pts.shape[0]):
        x, y, z = gauss_pts[i]
        ax.scatter(x, y, z, color='red', s=80, edgecolors='k', zorder=3)
        ax.text(x, y, z, f'G{i+1}', fontsize=12, ha='left', va='top', color='red', bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel("ξ (Natural Coordinate)")
    ax.set_ylabel("η (Natural Coordinate)")
    ax.set_zlabel("ζ (Natural Coordinate)")
    ax.set_title(f"Gauss Points and Element Nodes for {ele_type}")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    
    return

def gauss_pts_and_weights(ele_type, num_pts):
    """
    Retrieves the Gauss quadrature points and weights for a given element type and number of integration points.
    
    Parameters
    ----------
    ele_type : str
        The type of finite element. Supported types:
        - "D3_nn8_hex" : 8-node trilinear brick element
    num_pts : int
        The number of Gauss integration points.
        - Brick elements: Supports 1 or 8 points.
    
    Returns
    -------
    gauss_pts : np.ndarray of shape (2, num_pts)
        The Gauss quadrature points for the specified element type.
    gauss_weights : np.ndarray of shape (num_pts, 1)
        The corresponding Gauss quadrature weights.
    
    Raises
    ------
    ValueError
        If an unsupported element type is provided.
    """
    gauss_pts_all = {
        "D3_nn8_hex": di.gauss_points_3d_brick,
    }

    gauss_weights_all = {
        "D3_nn8_hex": di.gauss_weights_3d_brick,
    }
    
    if ele_type not in gauss_pts_all:
        raise ValueError(f"Unsupported element type: {ele_type}")
    
    gauss_pts = gauss_pts_all[ele_type](num_pts)
    gauss_weights = gauss_weights_all[ele_type](num_pts)
    
    return gauss_pts, gauss_weights

def plot_interpolate_field_natural_coords_single_element(fname: str, ele_type: str, node_values: np.ndarray, num_interp_pts: int = 10):
    """
    Plots a scalar field interpolated across a sampling of points in natural coordinates.
    Saves the file according to `fname`.
    Calls `interpolate_field_natural_coords_single_element` to perform interpolation.
    
    Parameters
    ----------
    fname : str
        The filename to save the plot.
    ele_type : str
        The type of finite element.
        - "D3_nn8_hex" : 8-node trilinear brick element
    node_values : numpy.ndarray of shape (n_nodes,)
        The values of the field at the element nodes.
    num_interp_pts : int, optional
        The number of interpolation points along each axis (default is 10).
    """
    # Define sampling points in natural coordinates    
    if ele_type in ["D3_nn8_hex"]:
        xi_vals = np.linspace(-1, 1, num_interp_pts)
        eta_vals = np.linspace(-1, 1, num_interp_pts)
        zeta_vals = np.linspace(-1, 1, num_interp_pts)
        XI, ETA, ZETA = np.meshgrid(xi_vals, eta_vals, zeta_vals)
        xi_filtered = XI.flatten()
        eta_filtered = ETA.flatten()
        zeta_filtered = ZETA.flatten()
        ref_nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]])
    else:
        raise ValueError(f"Unsupported element type: {ele_type}")
    
    # Compute interpolated field values
    interpolated_vals = interpolate_field_natural_coords_single_element(ele_type, node_values, xi_filtered, eta_filtered, zeta_filtered)
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xi_filtered, eta_filtered, zeta_filtered,
                    c=interpolated_vals.flatten(), cmap='coolwarm', edgecolors='k', s=30, alpha=0.8)
    fig.colorbar(sc, ax=ax, label='Interpolated Field')

    # Plot element edges (same structure, 3D version)
    cube_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in cube_edges:
        ax.plot([ref_nodes[edge[0], 0], ref_nodes[edge[1], 0]],
                [ref_nodes[edge[0], 1], ref_nodes[edge[1], 1]],
                [ref_nodes[edge[0], 2], ref_nodes[edge[1], 2]],
                'k-', lw=1)

    for i, (xi, eta, zeta) in enumerate(ref_nodes):
        ax.text(xi, eta, zeta, f'N{i+1}', fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6))

    ax.set_xlabel("ξ (Natural Coordinate)")
    ax.set_ylabel("η (Natural Coordinate)")
    ax.set_zlabel("ζ (Natural Coordinate)")
    ax.set_title(f"Interpolated Field for {ele_type}")
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

    return

def interpolate_field_natural_coords_single_element(ele_type, node_values, xi_vals, eta_vals, zeta_vals):
    """
    Interpolates a scalar field inside a single finite element using its shape functions
    in natural (reference) coordinates (ξ, η, ζ).

    Parameters
    ----------
    ele_type : str
        The type of finite element. Supported types:
        - "D3_nn8_hex" : 8-node trilinear brick element
    node_values : numpy.ndarray of shape (n_nodes,)
        The values of the field at the element nodes.
    xi_vals : numpy.ndarray of shape (n_xi,)
        The natural coordinate values (ξ) at which interpolation is performed.
    eta_vals : numpy.ndarray of shape (n_eta,)
        The natural coordinate values (η) at which interpolation is performed.
    zeta_vals : numpy.ndarray of shape (n_zeta,)
        The natural coordinate values (ζ) at which interpolation is performed.

    Returns
    -------
    interpolated_vals : numpy.ndarray of shape (n_xi, n_eta, n_zeta)
        The interpolated field values at the specified (ξ, η, ζ) points.

    Raises
    ------
    ValueError
        If an unsupported element type is provided.

    Notes
    -----
    - This function assumes that the element is in **natural coordinates** (ξ, η, ζ).
    - The function selects the appropriate shape function for the given element type.
    - Shape functions are evaluated at the given (ξ, η, ζ) values to interpolate the field.
    - Supports brick elements.
    """
    shape_function_map = {
        "D3_nn8_hex": di.D3_nn8_hex,
    }

    if ele_type not in shape_function_map:
        raise ValueError(f"Unsupported element type: {ele_type}")

    shape_function = shape_function_map[ele_type]

    interpolated_vals = np.zeros(len(xi_vals))
    for i in range(len(xi_vals)):
        xi = xi_vals[i]
        eta = eta_vals[i]
        zeta = zeta_vals[i]
        N = shape_function(np.array([xi, eta, zeta])).flatten() 
        interpolated_vals[i] = np.dot(N, node_values)

    return interpolated_vals.reshape((-1, 1))

def visualize_isoparametric_mapping_single_element(fname: str, ele_type, node_coords, node_values, num_interp_pts=20):
    """
    Visualizes the isoparametric mapping of a reference element to its physical shape.
    Calls `interpolate_field_natural_coords_single_element` to interpolate values inside the element.
    
    Parameters
    ----------
    fname : str
        The filename to save the plot.
    ele_type : str
        The type of finite element.
        - "D3_nn8_hex" : 8-node trilinear brick element
    node_coords : numpy.ndarray of shape (n_nodes, 3)
        The physical coordinates of the element nodes.
    node_values : numpy.ndarray of shape (n_nodes,)
        The values of the field at the element nodes.
    num_interp_pts : int, optional
        The number of interpolation points along each axis (default is 20 for smoother results).
    """ 

    # Define sampling points in natural coordinates
    if ele_type in ["D3_nn8_hex"]:
        xi_vals = np.linspace(-1, 1, num_interp_pts)
        eta_vals = np.linspace(-1, 1, num_interp_pts)
        zeta_vals = np.linspace(-1, 1, num_interp_pts)
        XI, ETA, ZETA = np.meshgrid(xi_vals, eta_vals, zeta_vals)
        xi_filtered = XI.flatten()
        eta_filtered = ETA.flatten()
        zeta_filtered = ZETA.flatten()
        ref_nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]])
    else:
        raise ValueError(f"Unsupported element type: {ele_type}")

    # Compute interpolated field values and mapped physical coordinates
    interpolated_vals = interpolate_field_natural_coords_single_element(ele_type, node_values, xi_filtered, eta_filtered, zeta_filtered).flatten()
    x_mapped = interpolate_field_natural_coords_single_element(ele_type, node_coords[:, 0], xi_filtered, eta_filtered, zeta_filtered).flatten()
    y_mapped = interpolate_field_natural_coords_single_element(ele_type, node_coords[:, 1], xi_filtered, eta_filtered, zeta_filtered).flatten()
    z_mapped = interpolate_field_natural_coords_single_element(ele_type, node_coords[:, 2], xi_filtered, eta_filtered, zeta_filtered).flatten()

    fig = plt.figure(figsize=(14, 6))
    ax_ref = fig.add_subplot(1, 2, 1, projection='3d')
    ax_phys = fig.add_subplot(1, 2, 2, projection='3d')

    # Plot in natural coordinates
    sc1 = ax_ref.scatter(xi_filtered, eta_filtered, zeta_filtered, c=interpolated_vals,
                         cmap='coolwarm', edgecolors='k', s=20, alpha=0.8)
    ax_ref.set_xlabel("ξ (Natural Coordinate)")
    ax_ref.set_ylabel("η (Natural Coordinate)")
    ax_ref.set_zlabel("ζ (Natural Coordinate)")
    ax_ref.set_title("Reference Element (Natural Coordinates)")
    fig.colorbar(sc1, ax=ax_ref, shrink=0.6, pad=0.1)

    for i, (xi, eta, zeta) in enumerate(ref_nodes):
        ax_ref.text(xi, eta, zeta, f'N{i+1}', fontsize=9, ha='center', va='center',
                    color='black', bbox=dict(facecolor='white', alpha=0.6))

    # Plot in physical coordinates
    sc2 = ax_phys.scatter(x_mapped, y_mapped, z_mapped, c=interpolated_vals,
                          cmap='coolwarm', edgecolors='k', s=20, alpha=0.8)
    ax_phys.set_xlabel("x (Physical Coordinate)")
    ax_phys.set_ylabel("y (Physical Coordinate)")
    ax_phys.set_zlabel("z (Physical Coordinate)")
    ax_phys.set_title("Mapped Element (Physical Coordinates)")
    fig.colorbar(sc2, ax=ax_phys, shrink=0.6, pad=0.1)

    for i, (x, y, z) in enumerate(node_coords):
        ax_phys.text(x, y, z, f'N{i+1}', fontsize=9, ha='center', va='center',
                     color='white', bbox=dict(facecolor='black', alpha=0.6))

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    return










def interpolate_gradient_natural_coords_single_element(ele_type, node_values, xi_vals, eta_vals, zeta_vals):
    """
    Interpolates the gradient of a scalar or vector field in natural coordinates (ξ, η, ζ).
    
    Parameters
    ----------
    ele_type : str
        The type of finite element.
    node_values : numpy.ndarray of shape (n_nodes,) or (n_nodes, 2)
        The values of the field at the element nodes.
        - If shape is (n_nodes,), the function interpolates a scalar field.
        - If shape is (n_nodes, 2), the function interpolates a vector field (e.g., displacement or velocity).
    xi_vals : numpy.ndarray of shape (n_xi,)
        The natural coordinate values (ξ) at which interpolation is performed.
    eta_vals : numpy.ndarray of shape (n_eta,)
        The natural coordinate values (η) at which interpolation is performed.
    zeta_vals : numpy.ndarray of shape (n_zeta,)
        The natural coordinate values (ζ) at which interpolation is performed.

    Returns
    -------
    gradient_natural : numpy.ndarray
        The interpolated field gradient in natural coordinates.
        - If interpolating a scalar field, the shape is (2, n_xi * n_eta * n_zeta).
        - If interpolating a vector field, the shape is (2, n_xi * n_eta * n_zeta, 2), where the last dimension corresponds
          to the two field components.
    """
    shape_function_derivatives = {
        "D3_nn8_hex": di.D3_nn8_hex_dxi,
    }

    if ele_type not in shape_function_derivatives:
        raise ValueError(f"Unsupported element type: {ele_type}")
    
    # Determine if the input is scalar or vector field
    scalar_field = len(node_values.shape) == 1 or node_values.shape[1] == 1
    n_field_components = 1 if scalar_field else node_values.shape[1]

    n_dim = 3  # assume 3D
    gradient_natural = np.zeros((n_dim, len(xi_vals), n_field_components))

    if scalar_field:
        node_values = node_values.flatten()

    for index in range(len(xi_vals)):
        xi = xi_vals[index]
        eta = eta_vals[index]
        zeta = zeta_vals[index]
        dN_dxi = shape_function_derivatives[ele_type](np.array([xi, eta, zeta]))

        if scalar_field:
            gradient_natural[:, index, 0] = dN_dxi.T @ node_values
        else:
            gradient_natural[:, index, :] = dN_dxi.T @ node_values

    return gradient_natural


def transform_gradient_to_physical(ele_type, node_coords, xi_vals, eta_vals, zeta_vals, gradient_natural):
    """
    Transforms the interpolated gradient from natural coordinates (ξ, η, ζ) to physical coordinates (x, y, z).

    Parameters
    ----------
    ele_type : str
        The type of finite element.
    node_coords : numpy.ndarray of shape (n_nodes, 3)
        The physical coordinates of the element nodes.
    xi_vals : numpy.ndarray
        The ξ natural coordinate values.
    eta_vals : numpy.ndarray
        The η natural coordinate values.
    zeta_vals : numpy.ndarray
        The ζ natural coordinate values.
    gradient_natural : numpy.ndarray of shape (3, n_points) or (3, n_points, n_components)
        The gradient in natural coordinates.

    Returns
    -------
    gradient_physical : numpy.ndarray of shape (3, n_points) or (3, n_points, n_components)
        The gradient transformed to physical coordinates.
    """
    is_scalar_field = len(gradient_natural.shape) == 2 or gradient_natural.shape[2] == 1
    n_components = 1 if is_scalar_field else gradient_natural.shape[2]

    n_points = len(xi_vals)
    gradient_physical = np.zeros_like(gradient_natural)

    for index in range(n_points):
        xi = xi_vals[index]
        eta = eta_vals[index]
        zeta = zeta_vals[index]

        J = compute_jacobian(ele_type, node_coords, xi, eta, zeta)
        J_inv = np.linalg.inv(J)

        if is_scalar_field:
            gradient_physical[:, index] = J_inv.T @ gradient_natural[:, index]
        else:
            for component in range(n_components):
                gradient_physical[:, index, component] = J_inv.T @ gradient_natural[:, index, component]

    return gradient_physical


def compute_jacobian(ele_type, node_coords, xi, eta, zeta):
    """
    Computes the Jacobian matrix for a 3D element type at natural coordinates (ξ, η, ζ).

    Parameters
    ----------
    ele_type : str
        The element type.
    node_coords : np.ndarray of shape (n_nodes, 3)
        The physical coordinates of the element nodes.
    xi, eta, zeta : float
        Natural coordinates.

    Returns
    -------
    J : np.ndarray of shape (3, 3)
        The Jacobian matrix at the point (ξ, η, ζ).
    """
    shape_function_derivatives = {
        "D3_nn8_hex": di.D3_nn8_hex_dxi,
    }

    if ele_type not in shape_function_derivatives:
        raise ValueError(f"Unsupported element type: {ele_type}")

    dN_dxi = shape_function_derivatives[ele_type](np.array([xi, eta, zeta]))
    J = node_coords.T @ dN_dxi

    return J







def compute_integral_of_derivative(ele_type, num_gauss_pts, node_coords, nodal_values,
                                      gauss_pts_and_weights, interpolate_gradient_natural_coords_single_element,
                                      transform_gradient_to_physical, compute_jacobian):
    gauss_pts, gauss_weights = gauss_pts_and_weights(ele_type, num_gauss_pts)

    is_vector_field = len(nodal_values.shape) == 2
    n_components = nodal_values.shape[1] if is_vector_field else 1
    integral = np.zeros((3, n_components)) if is_vector_field else np.zeros(3)

    for i in range(num_gauss_pts):
        xi, eta, zeta = gauss_pts[:, i]
        weight = gauss_weights[i, 0]

        gradient_natural = interpolate_gradient_natural_coords_single_element(
            ele_type, nodal_values, np.array([xi]), np.array([eta]), np.array([zeta])
        )

        gradient_physical = transform_gradient_to_physical(
            ele_type, node_coords, np.array([xi]), np.array([eta]), np.array([zeta]), gradient_natural
        )

        J = compute_jacobian(ele_type, node_coords, xi, eta, zeta)
        det_J = np.linalg.det(J)

        if is_vector_field:
            integral += weight * gradient_physical[:, 0, :] * det_J
        else:
            integral += weight * gradient_physical[:, 0] * det_J

    return integral
