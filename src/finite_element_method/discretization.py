import numpy as np


###########################################################
# ELEMENT TYPE INFORMATION -- WRAPPER FUNCTION
###########################################################

def element_info(ele_type: str):
    """
    Returns the number of coordinates, number of degrees of freedom (DOFs),
    and number of element nodes for a given finite element type.

    Parameters:
        ele_type (str): The element type identifier.

    Returns:
        tuple:
            - int: Number of coordinates (1 for 1D, 2 for 2D, 3 for 3D).
            - int: Number of degrees of freedom (same as number of coordinates).
            - int: Number of element nodes.

    Raises:
        ValueError: If ele_type is not recognized.
    """
    element_data = {
        "D3_nn8_hex": (3, 3, 8),  # 3D brick element with 8 nodes
    }

    if ele_type not in element_data:
        raise ValueError(f"Unknown element type: {ele_type}")

    return element_data[ele_type]


###########################################################
# SHAPE FUNCTIONS AND DERIVATIVES -- WRAPPER FUNCTIONS
###########################################################

def shape_fcn(ele_type: str, xi: np.ndarray) -> np.ndarray:
    """
    Evaluate the shape functions for a given finite element type at natural coordinates.

    Parameters
    ----------
    ele_type : str
        The element type identifier. Supported types include:
        - "D3_nn8_hex" : 3D brick element (8 nodes)

    xi : np.ndarray
        A NumPy array representing the natural coordinates where the shape
        functions should be evaluated.

    Returns
    -------
    N : np.ndarray
        A NumPy array containing the evaluated shape functions at xi.

    Raises
    ------
    ValueError
        If the element type is not recognized.

    Notes
    -----
    - This function provides a clean interface for evaluating shape functions
      without needing to call individual shape function implementations manually.
    - It is used in **finite element analysis (FEA)** for interpolation and
      numerical integration.
    """
    
    # Dictionary mapping element types to shape function implementations
    shape_function_map = {
        "D3_nn8_hex": D3_nn8_hex,
    }

    # Ensure the element type is valid
    if ele_type not in shape_function_map:
        raise ValueError(f"Unsupported element type '{ele_type}'. "
                         "Supported types: " + ", ".join(shape_function_map.keys()))

    # Evaluate the shape function for the given element type
    return shape_function_map[ele_type](xi)


###########################################################
# SHAPE FUNCTIONS AND DERIVATIVES -- WRAPPER FUNCTIONS
###########################################################

def shape_fcn(ele_type: str, xi: np.ndarray) -> np.ndarray:
    """
    Evaluate the shape functions for a given finite element type at natural coordinates.

    Parameters
    ----------
    ele_type : str
        The element type identifier. Supported types include:
        - "D3_nn8_hex" : 3D brick element (8 nodes)

    xi : np.ndarray
        A NumPy array representing the natural coordinates where the shape
        functions should be evaluated.

    Returns
    -------
    N : np.ndarray
        A NumPy array containing the evaluated shape functions at xi.

    Raises
    ------
    ValueError
        If the element type is not recognized.

    Notes
    -----
    - This function provides a clean interface for evaluating shape functions
      without needing to call individual shape function implementations manually.
    - It is used in **finite element analysis (FEA)** for interpolation and
      numerical integration.
    """
    
    # Dictionary mapping element types to shape function implementations
    shape_function_map = {
        "D3_nn8_hex": D3_nn8_hex,
    }

    # Ensure the element type is valid
    if ele_type not in shape_function_map:
        raise ValueError(f"Unsupported element type '{ele_type}'. "
                         "Supported types: " + ", ".join(shape_function_map.keys()))

    # Evaluate the shape function for the given element type
    return shape_function_map[ele_type](xi)


def shape_fcn_derivative(ele_type: str, xi: np.ndarray) -> np.ndarray:
    """
    Evaluate the shape function derivatives for a given finite element type at natural coordinates.

    Parameters
    ----------
    ele_type : str
        The element type identifier. Supported types include:
        - "D3_nn8_hex" : 3D brick element (8 nodes)

    xi : np.ndarray
        A NumPy array representing the natural coordinates where the shape
        function derivatives should be evaluated.

    Returns
    -------
    dN_dxi : np.ndarray
        A NumPy array containing the evaluated shape function derivatives at xi.
        - Each **row** corresponds to a **node**.
        - Each **column** corresponds to derivatives with respect to **ξ (0), η (1), and ζ (2)**.

    Raises
    ------
    ValueError
        If the element type is not recognized.

    Notes
    -----
    - This function provides a clean interface for evaluating shape function derivatives
      without needing to call individual implementations manually.
    - It is used in **finite element analysis (FEA)** for constructing the **B-matrix**,
      which relates strain to nodal displacements.
    """
    
    # Dictionary mapping element types to shape function derivative implementations
    shape_function_derivative_map = {
        "D3_nn8_hex": D3_nn8_hex_dxi,
    }

    # Ensure the element type is valid
    if ele_type not in shape_function_derivative_map:
        raise ValueError(f"Unsupported element type '{ele_type}'. "
                         "Supported types: " + ", ".join(shape_function_derivative_map.keys()))

    # Evaluate the shape function derivative for the given element type
    return shape_function_derivative_map[ele_type](xi)


###########################################################
# GAUSSIAN INTEGRATION POINTS AND WEIGHTS -- WRAPPER FUNCTION
###########################################################


def integration_info(ele_type: str):
    """
    Returns the number of integration points, integration points, and integration weights
    for a given finite element type.

    Parameters:
        ele_type (str): The element type identifier.

    Returns:
        tuple:
            - int: Number of integration points.
            - np.ndarray: Integration points (shape depends on element type).
            - np.ndarray: Integration weights (num_pts, 1).

    Raises:
        ValueError: If ele_type is not recognized.
    """
    element_data = {
        "D3_nn8_hex": (8, gauss_points_3d_brick(8), gauss_weights_3d_brick(8))
    }

    if ele_type not in element_data:
        raise ValueError(f"Unknown element type: {ele_type}")

    return element_data[ele_type]















###########################################################
###########################################################
# SHAPE FUNCTIONS AND DERIVATIVES -- 3D
###########################################################
###########################################################

def D3_nn8_hex(xi: np.ndarray) -> np.ndarray:
    """
    Compute the 3D trilinear shape functions for an eight-node trilinear brick element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (3,)) representing the natural coordinates (ξ, η, ζ).
        ξ, η and ζ are in the range [-1,1], defining the local brick coordinate system.

    Returns
    -------
    N : np.ndarray
        A (8,1) NumPy array containing the evaluated trilinear shape functions at (ξ, η, ζ).

    Notes
    -----
    - D3 refers to a **3D** element.
    - nn8 refers to **8 nodal values** (trilinear brick element).
    - The bilinear shape functions are:
        N1(ξ, η, ζ) = 0.125 * (1 - ξ) * (1 - η) * (1 - ζ)
        N2(ξ, η, ζ) = 0.125 * (1 + ξ) * (1 - η) * (1 - ζ)
        N3(ξ, η, ζ) = 0.125 * (1 + ξ) * (1 - η) * (1 + ζ)
        N4(ξ, η, ζ) = 0.125 * (1 - ξ) * (1 - η) * (1 + ζ)
        N5(ξ, η, ζ) = 0.125 * (1 - ξ) * (1 + η) * (1 - ζ)
        N6(ξ, η, ζ) = 0.125 * (1 + ξ) * (1 + η) * (1 - ζ)
        N7(ξ, η, ζ) = 0.125 * (1 + ξ) * (1 + η) * (1 + ζ)
        N8(ξ, η, ζ) = 0.125 * (1 - ξ) * (1 + η) * (1 + ζ)
    - These shape functions are used in **finite element analysis (FEA)**
      to interpolate field variables within a trilinear brick element.
    """
    N = np.zeros((8, 1))
    N[0, 0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])  # N1
    N[1, 0] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])  # N2
    N[2, 0] = 0.125 * (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])  # N3
    N[3, 0] = 0.125 * (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])  # N4
    N[4, 0] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])  # N5
    N[5, 0] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])  # N6
    N[6, 0] = 0.125 * (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])  # N7
    N[7, 0] = 0.125 * (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])  # N8
   
    return N


def D3_nn8_hex_dxi(xi: np.ndarray) -> np.ndarray:
    """
    Compute the derivatives of the 3D trilinear shape functions for an eight-node brick element.

    Parameters
    ----------
    xi : np.ndarray
        A 1D NumPy array (shape: (3,)) representing the natural coordinates (ξ, η, ζ).
        ξ, η and ζ are in the range [-1,1], defining the local brick coordinate system.

    Returns
    -------
    dN_dxi : np.ndarray
        A (8,3) NumPy array containing the derivatives of the shape functions
        with respect to the natural coordinates (ξ, η, ζ).
        Each row corresponds to a node, and columns correspond to derivatives
        with respect to ξ, η and ζ.

    Notes
    -----
    - D3 refers to a **3D** element.
    - nn8 refers to **8 nodal values** (trilinear brick element).
    - The derivatives of the trilinear shape functions are:
        dN1/dξ = -0.125 * (1 - η) * (1 - ζ), dN1/dη = -0.125 * (1 - ξ) * (1 - ζ), dN1/dζ = -0.125 * (1 - ξ) * (1 - η)
        dN2/dξ =  0.125 * (1 - η) * (1 - ζ), dN2/dη = -0.125 * (1 + ξ) * (1 - ζ), dN2/dζ = -0.125 * (1 + ξ) * (1 - η)
        dN3/dξ =  0.125 * (1 - η) * (1 + ζ), dN3/dη = -0.125 * (1 + ξ) * (1 + ζ), dN3/dζ =  0.125 * (1 + ξ) * (1 - η)
        dN4/dξ = -0.125 * (1 - η) * (1 + ζ), dN4/dη = -0.125 * (1 - ξ) * (1 + ζ), dN4/dζ =  0.125 * (1 - ξ) * (1 - η)
        dN5/dξ = -0.125 * (1 + η) * (1 - ζ), dN5/dη =  0.125 * (1 - ξ) * (1 - ζ), dN5/dζ = -0.125 * (1 - ξ) * (1 + η)
        dN6/dξ =  0.125 * (1 + η) * (1 - ζ), dN6/dη =  0.125 * (1 + ξ) * (1 - ζ), dN6/dζ = -0.125 * (1 + ξ) * (1 + η)
        dN7/dξ =  0.125 * (1 + η) * (1 + ζ), dN7/dη =  0.125 * (1 + ξ) * (1 + ζ), dN7/dζ =  0.125 * (1 + ξ) * (1 + η)
        dN8/dξ = -0.125 * (1 + η) * (1 + ζ), dN8/dη =  0.125 * (1 - ξ) * (1 + ζ), dN8/dζ =  0.125 * (1 - ξ) * (1 + η)

       
    - These derivatives are used in **finite element analysis (FEA)**
      to compute the strain-displacement matrix (B-matrix).
    """
    dN_dxi = np.zeros((8, 3))

    dN_dxi[0, 0] = -0.125 * (1.0 - xi[1]) * (1.0 - xi[2])   # dN1/dξ
    dN_dxi[0, 1] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[2])   # dN1/dη
    dN_dxi[0, 2] = -0.125 * (1.0 - xi[0]) * (1.0 - xi[1])   # dN1/dζ

    dN_dxi[1, 0] =  0.125 * (1.0 - xi[1]) * (1.0 - xi[2])   # dN2/dξ
    dN_dxi[1, 1] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[2])   # dN2/dη
    dN_dxi[1, 2] = -0.125 * (1.0 + xi[0]) * (1.0 - xi[1])   # dN2/dζ

    dN_dxi[2, 0] =  0.125 * (1.0 - xi[1]) * (1.0 + xi[2])   # dN3/dξ
    dN_dxi[2, 1] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[2])   # dN3/dη
    dN_dxi[2, 2] =  0.125 * (1.0 + xi[0]) * (1.0 - xi[1])   # dN3/dζ

    dN_dxi[3, 0] = -0.125 * (1.0 - xi[1]) * (1.0 + xi[2])   # dN4/dξ
    dN_dxi[3, 1] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[2])   # dN4/dη
    dN_dxi[3, 2] =  0.125 * (1.0 - xi[0]) * (1.0 - xi[1])   # dN4/dζ

    dN_dxi[4, 0] = -0.125 * (1.0 + xi[1]) * (1.0 - xi[2])   # dN5/dξ
    dN_dxi[4, 1] =  0.125 * (1.0 - xi[0]) * (1.0 - xi[2])   # dN5/dη
    dN_dxi[4, 2] = -0.125 * (1.0 - xi[0]) * (1.0 + xi[1])   # dN5/dζ

    dN_dxi[5, 0] =  0.125 * (1.0 + xi[1]) * (1.0 - xi[2])   # dN6/dξ
    dN_dxi[5, 1] =  0.125 * (1.0 + xi[0]) * (1.0 - xi[2])   # dN6/dη
    dN_dxi[5, 2] = -0.125 * (1.0 + xi[0]) * (1.0 + xi[1])   # dN6/dζ

    dN_dxi[6, 0] =  0.125 * (1.0 + xi[1]) * (1.0 + xi[2])   # dN7/dξ
    dN_dxi[6, 1] =  0.125 * (1.0 + xi[0]) * (1.0 + xi[2])   # dN7/dη
    dN_dxi[6, 2] =  0.125 * (1.0 + xi[0]) * (1.0 + xi[1])   # dN7/dζ

    dN_dxi[7, 0] = -0.125 * (1.0 + xi[1]) * (1.0 + xi[2])   # dN8/dξ
    dN_dxi[7, 1] =  0.125 * (1.0 - xi[0]) * (1.0 + xi[2])   # dN8/dη
    dN_dxi[7, 2] =  0.125 * (1.0 - xi[0]) * (1.0 + xi[1])   # dN8/dζ

    return dN_dxi


###########################################################
# GAUSSIAN INTEGRATION POINTS
###########################################################


def gauss_points_3d_brick(num_pts: int) -> np.ndarray:
    """
    Returns the Gauss-Legendre integration points for brick elements in 3D.

    Gauss integration points are used for numerical integration in finite element
    analysis. This function provides standard 3D Gauss points for a brick
    reference element.

    Available quadrature rules:
    - 1-point rule (center of the element).
    - 8-point rule (suitable for trilinear functions).

    Parameters:
        num_pts (int): Number of Gauss integration points (1 or 8).

    Returns:
        np.ndarray: A (3, num_pts) array containing the Gauss integration points.

    Raises:
        ValueError: If num_pts is not 1 or 8.
    """
    if num_pts not in {1, 8}:
        raise ValueError("num_pts must be 1 or 8.")

    xi_array = np.zeros((3, num_pts))

    if num_pts == 1:
        # Single-point quadrature (center of the reference square)
        xi_array[:, 0] = [0.0, 0.0, 0.0]
    elif num_pts == 8:
        # Eight-point quadrature rule
        sqrt_3_inv = 1.0 / np.sqrt(3)
        xi_array[:, 0] = [-sqrt_3_inv, -sqrt_3_inv, -sqrt_3_inv]
        xi_array[:, 1] = [sqrt_3_inv, -sqrt_3_inv, -sqrt_3_inv]
        xi_array[:, 2] = [sqrt_3_inv,  sqrt_3_inv, -sqrt_3_inv]
        xi_array[:, 3] = [-sqrt_3_inv,  sqrt_3_inv, -sqrt_3_inv]
        xi_array[:, 4] = [-sqrt_3_inv, -sqrt_3_inv,  sqrt_3_inv]
        xi_array[:, 5] = [sqrt_3_inv, -sqrt_3_inv,  sqrt_3_inv]
        xi_array[:, 6] = [sqrt_3_inv,  sqrt_3_inv,  sqrt_3_inv]
        xi_array[:, 7] = [-sqrt_3_inv,  sqrt_3_inv,  sqrt_3_inv]

    return xi_array


###########################################################
# GAUSSIAN INTEGRATION WEIGHTS
###########################################################

def gauss_weights_3d_brick(num_pts: int) -> np.ndarray:
    """
    Returns the Gauss-Legendre integration weights for brick elements in 3D.

    Gauss integration weights are used for numerical integration in finite element
    analysis. This function provides standard 3D Gauss weights for a brick
    reference element.

    Available quadrature rules:
    - 1-point rule: w = [4.0] (suitable for linear functions).
    - 4-point rule: w = [1.0, 1.0, 1.0, 1.0] (suitable for trilinear functions).

    Parameters:
        num_pts (int): Number of Gauss integration points (1 or 9).

    Returns:
        np.ndarray: A (num_pts, 1) array containing the Gauss integration weights.

    Raises:
        ValueError: If num_pts is not 1 or 8.
    """
    if num_pts not in {1, 8}:
        raise ValueError("num_pts must be 1 or 8.")

    if num_pts == 1:
        w_array = np.array([[8.0]])
    elif num_pts == 8:
        w_array = np.ones((8, 1))

    return w_array
