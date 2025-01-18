
import warnings
from StormReactor._standardization import _standardize_method, _standardize_parameters, _standardize_element, _standardize_element_type, _standardize_pollutants, _standardize_methods
from StormReactor.defs.ElementType import ElementType

class wqConfig:
    def __init__(self, element_id:str, element_type:str, pollutant:str, method:str, parameters:dict, model=None):
        self.element_id = element_id
        self.element_type = element_type
        self.pollutant = pollutant
        self.method = method
        self.parameters = parameters
        
        self._standardize_element_type = _standardize_element_type
        self._standardize_methods = _standardize_methods
        self._standardize_element = _standardize_element
        self._standardize_pollutants = _standardize_pollutants

        self.element_type = self._standardize_element_type(self.element_type)
        self._standardize_methods(self.method)

        if model:
            self._standardize_element(self.element_id, model)
            self._standardize_pollutants(self.pollutant, model)
        else:
            warnings.warn("Model not provided to config. Element and pollutant validation will not be performed.")

    def get_config(self):
        return self.config