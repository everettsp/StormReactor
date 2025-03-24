import warnings
from StormReactor.defs import REQUIRED_PARAMETERS, METHODS
from StormReactor.defs.ElementType import ElementType
import numpy as np
from collections import Counter

def _count_ids(ids:list):
    return Counter(ids)

def _standardize_config(config, model):
    """
    standardize the configuration for the water quality model
    sorts WQ configurations by priority and normalizes the priority values
    checks for duplicate priorities for the same node

    :param config: list, list of WQConfig objects
    :type config: list
    :param model: Model, swmmio model
    :type model: Model
    :return: list of WQConfig objects
    :rtype: list
    """
    methods = np.unique([c.method for c in config])
    config = sorted(config, key=lambda c: c.priority)

    link_to_node = model.links().InletNode.to_dict()
    node_ids = [link_to_node[c.element_id] if c.element_type == ElementType.Links else c.element_id for c in config]
    priorities = [c.priority for c in config]

    for node_id in np.unique(node_ids):
        node_subset = np.array(priorities)[np.array(node_ids) == node_id]
        priorities_subset = np.array(priorities)[np.array(node_ids) == node_id]
        
        if len(np.unique(priorities_subset)) != len(priorities_subset):
            warnings.warn(f"Duplicate priorities found for node {node_id}; using arbitrary order")

    normalized_priority = np.argsort(priorities)
    for ii, _ in enumerate(config):
        config[ii].priority = normalized_priority[ii]


    return config
    #if len(methods) > 1:
        

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
    all_elements = list(model.nodes().index) + list(model.links().index) + list(model.inp.subcatchments.index)
    if element_id not in all_elements:
        raise ValueError(f"Element {element_id} not found in model")

def _standardize_element_type(element_type):
    if element_type in ["node","nodes"]:
        element_type = ElementType.Nodes
    elif element_type in ["link","links"]:
        element_type = ElementType.Links
    elif element_type in ["subcatchment","subcatchments"]:
        element_type = ElementType.Subcatchments
    else:
        raise ValueError(f"Element type {element_type} not available. Must be 'node', 'link', or 'subcatchment'")

    return element_type

def _standardize_pollutants(pollutant, model):
    all_pollutants = list(model.inp.pollutants.index)
    if pollutant not in all_pollutants:
        raise ValueError(f"Pollutant {pollutant} not found in model")
    