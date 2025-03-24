METHODS = ["EventMeanConc",
           "ConstantRemoval",
           "CoRemoval",
           "ConcDependRemoval",
           "NthOrderReaction",
           "kCModel",
           "GravitySettling",
           "Erosion",
           "CSTR",
           "Phosphorus",
           "SewageFlux",
           "ViralDecay",
           "ConstantWasteLoad",
           "DryWeatherLoading",
           "CustomPollutProfile",
           ]

#TODO: might be cleaner to switch these this to a class

UNIVERAL_PARAMETERS = ["method"]

REQUIRED_PARAMETERS = {}
REQUIRED_PARAMETERS["EventMeanConc"] = ["C"]
REQUIRED_PARAMETERS["CoRemoval"] = ["R1","R2"]
REQUIRED_PARAMETERS["ConcDependRemoval"] = ["BC","R_l","R_u"]
REQUIRED_PARAMETERS["NthOrderReaction"] = ["k","n"]
REQUIRED_PARAMETERS["kCModel"] = ["k","C_s"]
REQUIRED_PARAMETERS["GravitySettling"] = ["k","C_s"]
REQUIRED_PARAMETERS["SewageFlux"] = ["Qhalf","v_sett","Smax","Resus_max","n"]
REQUIRED_PARAMETERS["ViralDecay"] = ["n","k","copollutant"]
REQUIRED_PARAMETERS["ConstantWasteLoad"] = ["L"]
REQUIRED_PARAMETERS["DryWeatherLoading"] = ["multiplier"]
REQUIRED_PARAMETERS["CustomPollutProfile"] = ["filename"]

REQUIRED_PARAMETERS = {key: UNIVERAL_PARAMETERS + value for key, value in REQUIRED_PARAMETERS.items()}