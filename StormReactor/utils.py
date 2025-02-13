
import numpy as np
from swmmio import Model
from datetime import datetime


def get_conduit_volume(model:Model, id:str=None):
    """
    get the conduit volume

    :param mdl: swmmio.Model, swmm model object
    :param id: str, conduit id
    :return: conduit volume or dict of all conduit volumes
    :rtype: float or dict
    """
    xsections = get_conduit_xsection(model)
    volumes = {c: xsections[c] * model.inp.conduits.loc[c,"Length"] for c in model.inp.conduits.index}

    if id is not None:
        return volumes[id]
    else:
        return volumes

def get_conduit_xsection(model:Model, id:str=None):
    """
    get the conduit cross-sectional area

    :param mdl: swmmio.Model, swmm model object
    :param id: str, conduit id
    :return: conduit cross-sectional area or dict of all conduit cross-sectional areas
    :rtype: float or dict 
    """

    xsections = {c:_calc_xsection_area(
        shape=model.inp.xsections.loc[c,"Shape"],
        diameter=model.inp.xsections.loc[c,"Geom1"],
        width=model.inp.xsections.loc[c,"Geom1"],
        height=model.inp.xsections.loc[c,"Geom2"],
        ) for c in model.inp.conduits.index}
    
    if id is not None:
        return xsections[id]
    else:
        return xsections
    

def _calc_xsection_area(shape:str, width:float=None, height:float=None, diameter:float=None) -> float:
    """
    get the cross-sectional area of a conduit

    :param shape: str, shape of the conduit (circular or rectangular)
    :param width: float, width of the conduit (for rectangular shape)
    :param height: float, height of the conduit (for rectangular shape)
    :param diameter: float, diameter of the conduit (for circular shape)
    :return: cross-sectional area of the conduit
    :rtype: float
    """

    if shape.upper() in ["CIRCULAR", "CIRCLE", "CIR"]:
        if diameter is None:
            raise ValueError("Diameter must be specified for a circular shape.")
        return np.pi * (diameter/2)**2

    elif shape in ["RECTANGULAR", "SQUARE", "REC"]:
        if width is None or height is None:
            raise ValueError("Width and height must be specified for a rectangular shape.")
        return width * height

    else:
        raise ValueError("Invalid shape specified.")
    

def calc_surface_area(shape:str, length:float, depth:float, width:float=None, height:float=None, diameter:float=None) -> float:
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



# load time patterns
from datetime import datetime
def get_patterns_as_df(mdl):
    """
    retrieve time patterns from a swmm model and return them as a time indexed pandas dataframe

    :param mdl: SWMM Model
    :type mdl: swmmio.Model
    :return: time indexed pandas dataframe
    :rtype: pandas.DataFrame   
    
    """
    time_array = [datetime.strptime(f"{hour}:00", "%H:%M").time() for hour in range(24)]
    
    patterns_df = mdl.inp.patterns.T

    patterns_df = patterns_df.drop(index=['Type'])
    if len(patterns_df) != 24:
        raise NotImplementedError("Time patterns must be defined for all 24 hours of the day")
    
    patterns_df.index = time_array
    return patterns_df

def is_weekend(ct: datetime) -> bool:
    """
    determine if a given datetime is a weekend

    :param ct: datetime, current datetime
    :type ct: datetime
    :return: True if weekend, False if not
    :rtype: bool
    """

    is_weekend = ct.weekday() == 5 or ct.weekday() == 6
    return is_weekend


from StormReactor.WaterQualityCaches import _CreateDryWeatherLoadingCache

def get_dwf_load(element:str, pollutant:str, datestamp:datetime, dt:int, cache:dict=None):
    # determine which pattern to use
    if is_weekend(datestamp):
        pattern_dict = cache["weekend_pattern_dict"]
    else:
        pattern_dict = cache["weekday_pattern_dict"]

    patterns = cache["patterns"]

    # get the pattern at the node
    pattern_name = pattern_dict[element]

    # get the dwf rate corresponding to the pattern and time of day
    average_value = cache["dwf"].loc[element, "AverageValue"]
    dwf_rate = average_value * patterns.loc[patterns.index.asof(datestamp.time()), pattern_name]


    


    # calculate dwf volume [m^3], concentration [mg/L], and load [mg]
    dwf_volume = dwf_rate * dt
    dwf_conc = cache["pollutants"].loc[pollutant, "DWFConcen"]
    dwf_load = dwf_volume * dwf_conc

    return dwf_load
