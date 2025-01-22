
#TODO: finish adding units to the docstrings, double check parameter names and types


class WQParams(dict):
    def __init__(self) -> None:
        super().__init__()

    def EventMeanConc(self, C: float) -> None:
        """
        Sets the parameters for an event mean concentration.

        :param C: float, event mean concentration
        :type C: float
        """
        self["C"] = C

    def CoRemoval(self, R1: float, R2: float) -> None:
        """
        Sets the parameters for a constant removal rate.

        :param R1: float, removal rate 1
        :type R1: float
        :param R2: float, removal rate 2
        :type R2: float
        
        """
        self["R1"] = R1
        self["R2"] = R2

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
        self["BC"] = BC
        self["R_l"] = R_l
        self["R_u"] = R_u

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
        self["k"] = k
        self["n"] = n

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

        self["k"] = k
        self["C_s"] = C_s

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

        self["k"] = k
        self["C_s"] = C_s

    def SewageFlux(self, Qhalf: float, v_sett: float, Smax: float) -> None:
        """
        Sets the sewage flux parameters.
        :param Qhalf: The flow rate at half saturation (m^3/s).
        :type Qhalf: float
        :param v_sett: The settling velocity (m/s).
        :type v_sett: float
        :param Smax: The maximum storage (kg).
        :type Smax: float
        :return: None
        """

        self["Qhalf"] = Qhalf
        self["v_sett"] = v_sett
        self["Smax"] = Smax