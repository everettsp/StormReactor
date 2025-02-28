import numpy as np
import pandas as pd

class WQStorage:
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
