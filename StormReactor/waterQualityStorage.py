import numpy as np
import pandas as pd

class WQStorage:
    """
    WQStorage is a class designed to manage water quality data for a set of elements and pollutants. 
    It provides storage, flux, and state management for each combination of element and pollutant.
    Attributes:
        element_ids (list): A list of element IDs (e.g., nodes or links in a network).
        pollutants (list): A list of pollutant names or identifiers.
        flag (int): An integer flag indicating the type of elements (0 for nodes, 1 for links). Default is 0.
        df (pd.DataFrame): A DataFrame storing the water quality values for each element and pollutant.
        fluxes (pd.DataFrame): A DataFrame storing the flux values for each element and pollutant.
        states (pd.DataFrame): A DataFrame storing the state values for each element and pollutant, initialized to NaN.
    Methods:
        _get_storage(element_id, pollutant):
            Retrieves the storage value for a specific element and pollutant.
            Args:
                element_id: The ID of the element.
                pollutant: The name or identifier of the pollutant.
            Returns:
                float: The storage value.
        _set_storage(element_id, pollutant, value):
            Sets the storage value for a specific element and pollutant.
            Args:
                element_id: The ID of the element.
                pollutant: The name or identifier of the pollutant.
                value: The value to set.
        _get_flux(element_id, pollutant):
            Retrieves the flux value for a specific element and pollutant.
            Args:
                element_id: The ID of the element.
                pollutant: The name or identifier of the pollutant.
            Returns:
                float: The flux value.
        _set_flux(element_id, pollutant, value):
            Sets the flux value for a specific element and pollutant.
            Args:
                element_id: The ID of the element.
                pollutant: The name or identifier of the pollutant.
                value: The value to set.
        _get_state(element_id, pollutant):
            Retrieves the state value for a specific element and pollutant.
            Args:
                element_id: The ID of the element.
                pollutant: The name or identifier of the pollutant.
            Returns:
                float: The state value.
        _set_state(element_id, pollutant, value):
            Sets the state value for a specific element and pollutant.
            Args:
                element_id: The ID of the element.
                pollutant: The name or identifier of the pollutant.
                value: The value to set.
    """


    def __init__(self, element_ids, pollutants, flag:int=0):
        self.element_ids = element_ids
        self.pollutants = pollutants
        self.values = np.zeros(shape=[len(element_ids),len(pollutants)])
        self.flag = flag # 0 for nodes, 1 for links
        self.df = pd.DataFrame(self.values, index=self.element_ids, columns=self.pollutants, dtype=float).copy()

        self.fluxes = pd.DataFrame(self.values, index=self.element_ids, columns=self.pollutants, dtype=float).copy()

        self.states = pd.DataFrame(self.values, index=self.element_ids, columns=self.pollutants, dtype=float).copy()
        self.states[:] = np.nan

    def _get_storage(self, element_id, pollutant):
        return self.df.loc[element_id, pollutant]

    def _set_storage(self, element_id, pollutant, value):
        self.df.loc[element_id, pollutant] = value

    def _get_flux(self, element_id, pollutant):
        return self.fluxes.loc[element_id, pollutant]

    def _set_flux(self, element_id, pollutant, value):
        self.fluxes.loc[element_id, pollutant] = value

    def _get_state(self, element_id, pollutant):
        return self.states.loc[element_id, pollutant]
    
    def _set_state(self, element_id, pollutant, value):
        self.states.loc[element_id, pollutant] = value
