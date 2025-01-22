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
           ]


#TODO: might be cleaner to switch these this to a class



REQUIRED_PARAMETERS = {}
REQUIRED_PARAMETERS["EventMeanConc"] = ["C"]
REQUIRED_PARAMETERS["CoRemoval"] = ["R1","R2"]
REQUIRED_PARAMETERS["ConcDependRemoval"] = ["BC","R_l","R_u"]
REQUIRED_PARAMETERS["NthOrderReaction"] = ["k","n"]
REQUIRED_PARAMETERS["kCModel"] = ["k","C_s"]
REQUIRED_PARAMETERS["GravitySettling"] = ["k","C_s"]
REQUIRED_PARAMETERS["SewageFlux"] = ["Qhalf","v_sett","Smax"]

