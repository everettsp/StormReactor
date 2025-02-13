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
           "DryWeatherLoading"
           ]

#TODO: might be cleaner to switch these this to a class

REQUIRED_PARAMETERS = {}
REQUIRED_PARAMETERS["EventMeanConc"] = ["method","C"]
REQUIRED_PARAMETERS["CoRemoval"] = ["method","R1","R2"]
REQUIRED_PARAMETERS["ConcDependRemoval"] = ["method","BC","R_l","R_u"]
REQUIRED_PARAMETERS["NthOrderReaction"] = ["method","k","n"]
REQUIRED_PARAMETERS["kCModel"] = ["method","k","C_s"]
REQUIRED_PARAMETERS["GravitySettling"] = ["method","k","C_s"]
REQUIRED_PARAMETERS["SewageFlux"] = ["method","Qhalf","v_sett","Smax","Resus_max","n"]
REQUIRED_PARAMETERS["ViralDecay"] = ["method","n","k","copollutant"]
REQUIRED_PARAMETERS["ConstantWasteLoad"] = ["method","L"]
REQUIRED_PARAMETERS["DryWeatherLoading"] = ["method","multiplier"]
