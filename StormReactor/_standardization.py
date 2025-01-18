import warnings
from StormReactor.defs import REQUIRED_PARAMETERS, METHODS
from StormReactor.defs.ElementType import ElementType

def _standardize_method(method:str):
    if method not in METHODS:
        raise ValueError(f"Method {method} not found in methods list")

def _standardize_parameters(parameters, method:str):
    _standardize_method(method)

    for param in REQUIRED_PARAMETERS[method]:
        if param not in parameters:
            raise ValueError(f"Parameter {param} is missing for {method} method. Required parameters include: {REQUIRED_PARAMETERS[method]}")
    
    for param in parameters:
        if param not in REQUIRED_PARAMETERS[method]:
            warnings.warn(f"Parameter {param} is not a valid parameter for {method} method and will be ignored.")



def _standardize_element(element_id, model):
    all_elements = list(model.nodes().index) + list(model.links().index)
    if element_id not in all_elements:
        raise ValueError(f"Element {element_id} not found in model")

def _standardize_element_type(element_type):
    if element_type not in ["node", "link"]:
        raise ValueError(f"Element type {element_type} not available. Must be 'node' or 'link'")

    if element_type == "node":
        element_type = ElementType.Nodes
    else:
        element_type = ElementType.Links

    return element_type

def _standardize_pollutants(pollutant, model):
    all_pollutants = list(model.inp.pollutants.index)
    if pollutant not in all_pollutants:
        raise ValueError(f"Pollutant {pollutant} not found in model")
    
def _standardize_methods(method):
    if method not in METHODS:
        raise ValueError(f"Method {method} not found in methods list")
    