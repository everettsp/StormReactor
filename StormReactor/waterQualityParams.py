#TODO: finish adding units to the docstrings, double check parameter names and types

import warnings
import pandas as pd
from pathlib import Path


from StormReactor.defs.ElementType import ElementType

class WQParams(dict):
    def __init__(self) -> None:
        super().__init__()
        self.clear()

    def EventMeanConc(self, C: float) -> None:
        """
        Sets the parameters for an event mean concentration.

        :param C: float, event mean concentration
        :type C: float
        """
        self.clear()
        self["method"] = "EventMeanConc"
        self["C"] = C
        return self

    def CoRemoval(self, R1: float, R2: float) -> None:
        """
        Sets the parameters for a constant removal rate.

        :param R1: float, removal rate 1
        :type R1: float
        :param R2: float, removal rate 2
        :type R2: float
        
        """
        self.clear()
        self["method"] = "CoRemoval"
        self["R1"] = R1
        self["R2"] = R2
        return self

    def ConcDependRemoval(self, BC: float, R_l: float, R_u: float) -> None:
        """
        Sets the parameters for a concentration dependent removal.

        :param BC: float, breakpoint concentration
        :type BC: float
        :param R_l: float, removal rate lower
        :type R_l: float
        :param R_u: float, removal rate upper
        :type R_u: float
        """
        self.clear()
        self["method"] = "ConcDependRemoval"
        self["BC"] = BC
        self["R_l"] = R_l
        self["R_u"] = R_u
        return self

    def NthOrderReaction(self, k: float, n: float) -> None:
        """
        Sets the parameters for an Nth order reaction.

        :param k: The reaction rate constant.
        :type k: float
        :param n: The reaction order.
        :type n: float
        :return: None
        :rtype: None
        """
        self.clear()
        self["method"] = "NthOrderReaction"
        self["k"] = k
        self["n"] = n
        return self

    def kCModel(self, k: float, C_s: float) -> None:
        """
        Updates the model parameters with the given values.
        :param k: The rate constant.
        :type k: float
        :param C_s: The concentration of the substance.
        :type C_s: float
        :return: None
        :rtype: None
        """
        self.clear()
        self["method"] = "kCModel"
        self["k"] = k
        self["C_s"] = C_s
        return self

    def GravitySettling(self, k: float, C_s: float) -> None:
        """
        Calculate and set the parameters for gravity settling.
        :param k: Settling velocity constant.
        :type k: float
        :param C_s: Suspended solids concentration.
        :type C_s: float
        :return: None
        :rtype: None
        """
        self.clear()
        self["method"] = "GravitySettling"
        self["k"] = k
        self["C_s"] = C_s
        return self

    def SewageFlux(self, Qhalf: float, v_sett: float, Smax: float, Resus_max: float, n: float=2) -> None:
        """
        Sets the sewage flux parameters.
        :param Qhalf: The flow rate at half saturation (m^3/s).
        :type Qhalf: float
        :param v_sett: The settling velocity (m/s).
        :type v_sett: float
        :param Smax: The maximum storage (kg).
        :type Smax: float
        :param Resus_max: The maximum resuspension rate (kg/s).
        :type Resus_max: float
        :param n: Resuspension curve steepness.
        :type n: float
        :return: None
        """
        self.clear()
        self["method"] = "SewageFlux"
        self["Qhalf"] = Qhalf
        self["v_sett"] = v_sett
        self["Smax"] = Smax
        self["Resus_max"] = Resus_max
        self["n"] = n
        return self
    
    def ViralDecay(self, copollutant:str, k:float, n:int) -> None:
        """
        Sets the decay parameters.
        :param k: The decay rate constant.
        :type k: float
        :return: None
        """
        self.clear()
        self["method"] = "ViralDecay"
        self["copollutant"] = copollutant
        self["k"] = k
        self["n"] = n
        return self
    
    def ConstantWasteLoad(self, L:float) -> None:
        """
        Sets the parameters for a constant concentration.

        :param L: float, constant load
        :type L: float
        """
        self.clear()
        self["method"] = "ConstantWasteLoad"
        self["L"] = L
        return self
    
    def DryWeatherLoading(self, multiplier:float) -> None:
        """
        No parameters required for dry-weather loading (retrieved from SWMM model).
        """
        self.clear()
        self["method"] = "DryWeatherLoading"
        self["multiplier"] = multiplier
        return self
    

    def CustomPollutProfile(self, filename:str|Path) -> None:
        """        
        :param filename: str, name of the file containing the custom pollutant loading data
        :type filename: str
        """
        self.clear()
        self["method"] = "CustomPollutProfile"
        self["filename"] = filename
        return self


def _standardize_custom_daily_profile(custom_profile, model, element_type):
    if element_type is not ElementType.Subcatchments:
        raise ValueError(f"Invalid element_type: {element_type}. Must be one of: {ElementType.Subcatchments}")

    custom_profile.columns = [col.strip() for col in custom_profile.columns]
    
    if custom_profile.isna().sum().sum() > 0:
        raise ValueError("The viral_load_profiles dataframe contains NaN values.")
    
    timestep = (pd.to_datetime('2000-01-01 ' + str(custom_profile.index[1])) - pd.to_datetime('2000-01-01 ' + str(custom_profile.index[0])))
    if not timestep * custom_profile.shape[0] == pd.Timedelta('1 days 00:00:00'):
        raise ValueError("The viral_load_profiles dataframe does not have a 24-hour time range.")
    
    # compare loaded profile columns against model
    model_elements = getattr(model.inp, "subcatchments").index

    missing_elements = [element for element in model_elements if element not in custom_profile.columns.tolist()]
    if missing_elements:
        raise ValueError(f"The viral_load_profiles dataframe is missing the following elements: {missing_elements}")
    
    extra_elements = [element for element in custom_profile.columns if element not in model_elements]
    if extra_elements:
        warnings.warn(f"Warning: The viral_load_profiles dataframe contains the following extra elements: {extra_elements}")
    
    return custom_profile


def load_custom_daily_profile(file_path, element_type, model=None):
    custom_profile = pd.read_csv(file_path, index_col=0)
    
    custom_profile.index = pd.to_datetime(custom_profile.index, format='%H:%M').time
    
    _standardize_custom_daily_profile(custom_profile=custom_profile, model=model, element_type=element_type)
    return custom_profile

