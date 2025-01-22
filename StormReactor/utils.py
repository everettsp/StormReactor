
import numpy as np

def get_surface_area(shape, length, depth, width=None, height=None, diameter=None):
    """
    get the surface area (air interface) of flow in a conduit

    :param shape: str, shape of the conduit (circular or rectangular)
    :param length: float, length of the conduit
    :param depth: float, depth of the flow in the conduit
    :param width: float, width of the conduit (for rectangular shape)
    :param height: float, height of the conduit (for rectangular shape)
    :param diameter: float, diameter of the conduit (for circular shape)
    :return: surface area of the flow in the conduit
    :rtype: float
    """

    if shape.upper() in ["CIRCULAR", "CIRCLE", "CIR"]:
        if diameter is None:
            raise ValueError("Diameter must be specified for a circular shape.")
        
        radius = diameter / 2
        if depth > radius:
            theta = 2 * np.arccos((radius-depth)/radius)
            x = depth - radius
            fill_width = 2 * x * np.tan(theta/2)

        elif depth < radius:
            theta = 2 * np.arccos((radius-depth)/radius)
            x = diameter - radius - depth
            fill_width = 2 * x * np.tan(theta/2)
        elif depth == radius:
            fill_width = radius*2
        elif depth == diameter:
            fill_width = 0
        elif depth == 0:
            fill_width = 0
        else:
            raise ValueError(f"depth ({depth}) must be less than or equal to the diameter of the circle ({diameter}).")

    elif shape in ["RECTANGULAR", "SQUARE", "REC"]:
        fill_width = width

    else:
        raise ValueError("Invalid shape specified.")
    
    return length * fill_width
