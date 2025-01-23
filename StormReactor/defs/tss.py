from pathlib import Path
import pandas as pd
import numpy as np

TSS_CLASSES = pd.DataFrame(
    index=[f"tss{ii}" for ii in range(6)],
    data=np.array([[.2, 2, 10, 30, 125, 500], [0, .2, 2, 10, 30, 125]]).T,
    columns=["upper_particle_size(um)", "lower_particle_size(um)"]
)

TSS_CLASSES["settling_velocity(m/h)"] = [0.01, 0.4, 1.22, 3, 10, 17]
TSS_CLASSES["settling_velocity(m/s)"] = TSS_CLASSES["settling_velocity(m/h)"] / 3600
TSS_CLASSES["cummulitive_distribution_dw"] = np.array([0, 2, 14, 32, 74, 100])
TSS_CLASSES["cummulitive_distribution_ww"] = np.array([0, 4, 20, 50, 95, 100])


def disaggregate(tss_data:dict[pd.DataFrame]):
    """
    Applies the hard-coded fractal model to transform aggregated TSS into discrete classes

    PARAMETERS
        tss_data: dict[pd.DataFrame]
            Dictionary of TSS dataframes
    
    RETURNS
        tss: pd.DataFrame
            DataFrame with the disaggregated TSS data
    """
    dfs = []
    for node in tss_data.columns:
        obs_subset = np.matmul(tss_data.loc[:,[node]],TSS_CLASSES["particle_size_distribution"].values.reshape(1,-1))
        obs_subset.columns = pd.MultiIndex.from_product([[node], TSS_CLASSES.index])
        dfs.append(obs_subset)
    tss = pd.concat(dfs, axis=1)
    return tss


def cum_distirbution_to_distribution(xx: np.array) -> np.array:
    if xx[-1] == 100:
        xx = xx / 100
    xx = [xx[0]] + [x - xx[ii] for ii, x in enumerate(xx[1:])]
    return np.array(xx)

TSS_CLASSES["particle_size_distribution"] = cum_distirbution_to_distribution(TSS_CLASSES["cummulitive_distribution_dw"].values)
TSS_CLASSES["swmm_label"] = [f"POLLUT_CONC_{x}" for x in range(len(TSS_CLASSES))]

WWTP_TO_NODE = {"south": "node_13", "west": "node_23", "north": "node_3"}

def load_tss_data(data_dir: Path) -> pd.DataFrame:
    filename = data_dir / "TSS" / "City Of Winnipeg Treatment Plant TSS Data.xlsx"
    dfs = {}
    sheet_names = {
        "south": "South End Plant",
        "north": "North End Plant",
        "west": "West End Plant"
    }
    for key in sheet_names:
        df = pd.read_excel(filename, header=[2], index_col=0, date_format="YYYY-MM-DD", sheet_name=sheet_names[key])
        df.columns = [col.strip() for col in df.columns]
        df.columns = ["flow(ML/d)", "TSS(ppm)"]
        df.loc[df["TSS(ppm)"] == 0, "TSS(ppm)"] = np.nan
        water_rho = 1000  # g/L
        df["TSS(mg/L)"] = df["TSS(ppm)"] / 1E6 * water_rho * 1E3  # g_solid/Mg_water / Mg_water/g_water * g_water/L_water * mg_solid/g_solid
        df["TSS(kg/d)"] = df["flow(ML/d)"] * 1E6 * df["TSS(mg/L)"] / 1E6
        """
        df = pd.DataFrame(
            data=np.matmul(df["raw (mg/L)"].to_numpy().reshape(-1, 1), TSS_CLASSES["particle_size_distribution"].values.reshape(1, -1)),
            columns=[f"{col}(mg/L)" for col in TSS_CLASSES.index]
        )
        """
        dfs[key] = df
    return dfs

def preprocess_tss_data(data_dir: Path) -> pd.DataFrame:
    """
    Preprocess the TSS data from the City
    
    PARAMETERS
    data_dir: Path
        Path to the data directory

    RETURNS
    df: pd.DataFrame
        DataFrame with the TSS data
    """
    tss_data = load_tss_data(data_dir)
    df = pd.DataFrame.from_dict({WWTP_TO_NODE[key]: tss_data[key]["TSS(mg/L)"] for key in tss_data})
    return df