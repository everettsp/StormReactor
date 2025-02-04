import numpy as np
import pandas as pd

class WQStorage:
    def __init__(self, element_ids, pollutants, flag:int=0):
        self.element_ids = element_ids
        self.pollutants = pollutants
        self.values = np.zeros(shape=[len(element_ids),len(pollutants)])
        self.flag = flag # 0 for nodes, 1 for links
        self.df = pd.DataFrame(self.values, index=self.element_ids, columns=self.pollutants)

    def _get_storage(self, element_id, pollutant):
        return self.df.loc[element_id, pollutant]

    def _set_storage(self, element_id, pollutant, value):
        self.df.loc[element_id, pollutant] = value