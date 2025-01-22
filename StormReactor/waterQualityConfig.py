
import warnings
from StormReactor._standardization import _standardize_method, _standardize_parameters, _standardize_parameters, _standardize_method, _standardize_element, _standardize_element_type, _standardize_pollutants
from StormReactor.defs.ElementType import ElementType
from swmmio import Model
#from StormReactor.WQParams import WQParams

class WQConfig:
    def __init__(self, element_id:str, element_type:str, pollutant:str, method:str, parameters:dict, model:Model=None):
        """
        configuration class for water quality class

        :param element_id: str, element id
        :type element_id: str
        :param element_type: str, element type
        :type element_type: str
        :param pollutant: str, pollutant name
        :type pollutant: str
        :param method: str, method name
        :type method: str
        :param parameters: dict, parameters for the method
        :type parameters: dict
        :param model: Model, swmmio model
        :type model: Model
        """
        
        self.element_id = element_id
        self.element_type = element_type
        self.pollutant = pollutant
        self.method = method
        self.parameters = parameters
        
        self._standardize_element_type = _standardize_element_type
        self._standardize_element = _standardize_element
        self._standardize_pollutants = _standardize_pollutants
        self._standardize_parameters = _standardize_parameters

        self.element_type = self._standardize_element_type(self.element_type)
        self._standardize_parameters(self.parameters, self.method)


        if model:
            self._standardize_element(self.element_id, model)
            self._standardize_pollutants(self.pollutant, model)
        else:
            warnings.warn("Model not provided to config. Element and pollutant validation will not be performed.")

    def get_config(self):
        return self.config