from StormReactor.utils import get_patterns_as_df
from swmmio import Model
from StormReactor.waterQualityParams import load_custom_daily_profile


def _CreateDryWeatherLoadingCache(model:Model):
# hash the node-pattern mappings for weekday/weekend
    cache = {}
    cache["weekday_pattern_dict"] = model.inp.dwf.iloc[:,2]
    cache["weekend_pattern_dict"] = model.inp.dwf.iloc[:,3]
    # load the patterns as pandas time-indexed dataframe
    cache["patterns"] = get_patterns_as_df(model)
    # load the pollutants
    cache["pollutants"] = model.inp.pollutants
    cache["dwf"] = model.inp.dwf
    return cache



#def _CreateCustomPollutProfileCache(model:Model):
#    cache = {}
#    cache["pollutants"] = load_custom_daily_profile(filename)
#    return cache