from pyswmm import Simulation, Nodes, Links
import pyswmm.toolkitapi as tka
import numpy as np
from scipy.integrate import ode

import pandas as pd
from tqdm import tqdm


from StormReactor.waterQualityStorage import WQStorage
from StormReactor.waterQualityConfig import WQConfig
from StormReactor._standardization import _standardize_method, _standardize_parameters, _standardize_config


from swmmio import Model

import warnings
from StormReactor.defs.ElementType import ElementType
from StormReactor.utils import get_dwf_load
from StormReactor.WaterQualityCaches import _CreateDryWeatherLoadingCache
from StormReactor.waterQualityParams import load_custom_daily_profile

# List of Exception Classes
class PySWMMStepAdvanceNotSupported(Exception):
    """
    Exception raised for impossible network trace.
    """
    def __init__(self):
        self.message = "PySWMM sim.step_advance() feature is currently unsuppprted."
        super().__init__(self.message)

class waterQuality:
    """
    Water quality module for SWMM

    This class provides all the necessary code to run StormReactor's
    water quality module with a SWMM simulation.

    Attributes
    __________
    config : dict
        dictionary with node/links where water quality methods are to
        be simulated, the pollutants to simulate, the pollutant water
        quality method to simulate, and the method's parameters.

        example:
        config = {
            '11': {'type': 'node', 'pollutant': 'P1', 'method': 'EventMeanConc', 'parameters': {"C": 10}},
            '5': {'type': 'node', 'pollutant': 'P1', 'method': 'ConstantRemoval', 'parameters': {"R": 5}},
            'Link1': {'type': 'link', 'pollutant': 'P1', 'method': 'EventMeanConc', 'parameters': {"C": 10}},
            'Link2': {'type': 'link', 'pollutant': 'P1', 'method': 'ConstantRemoval', 'parameters': {"R": 5}}
            }

    Methods
    _______
    updateWQState
        Updates the pollutant concentration during a SWMM simulation for
        all methods except CSTR.

    updateCSTRWQState
        Updates the pollutant concentration during a SWMM simulation for
        a CSTR.
    """

    # Initialize class
    def __init__(self, sim, config:list[WQConfig]):

        config = _standardize_config(config, model=Model(sim._model.inpfile))

        self.sim = sim
        self.config = config
        self.start_time = self.sim.start_time
        self.last_timestep = self.start_time
        self.solver = ode(self._CSTR_tank)
        self.model = Model(self.sim._model.inpfile)

        # Create cache for water quality methods
        # This gets calculated once per simulation (so it doesn't have to be recalculated at every simulation step)
        self._CreateCache()

        self.storage = WQStorage(
            element_ids = self.model.nodes().index.to_list(),
            pollutants = np.unique([c.pollutant for c in self.config]))
            #element_ids = np.unique([c.element_id for c in self.config]),
    
        # Water quality methods
        self.method = {
            "EventMeanConc": self._EventMeanConc,
            "ConstantRemoval": self._ConstantRemoval,
            "CoRemoval": self._CoRemoval,
            "ConcDependRemoval": self._ConcDependRemoval,
            "NthOrderReaction": self._NthOrderReaction,
            "kCModel": self._kCModel,
            "GravitySettling": self._GravitySettling,
            #"Erosion": self._Erosion,
            "CSTR": self._CSTRSolver,
            "Phosphorus": self._Phosphorus,
            "SewageFlux": self._SewageFlux,
            "ViralDecay": self._ViralDecay,
            "ConstantWasteLoad": self._ConstantWasteLoad,
            "DryWeatherLoading": self._DryWeatherLoading,
            "CustomPollutProfile": self._CustomPollutProfile
            }

        self.node_to_link = self._node_to_link()
        self.link_to_node = self._link_to_node()
        self.subcatchment_to_node = self._subcatchment_to_node()

    def _link_to_node(self):
        return self.model.links().OutletNode.to_dict()

    def _node_to_link(self):
        return {row.InletNode: link for link, row in self.model.links().iterrows()}
    
    def _subcatchment_to_node(self):
        return self.model.subcatchments().Outlet.to_dict()
    
    def split_ids(self, id, element_type):
        if element_type == ElementType.Nodes:
                node_id = id
                link_id = self.node_to_link[node_id]
        elif element_type == ElementType.Links:
            link_id = id
            node_id = self.node_to_link[link_id]
        else:
            raise ValueError(f"Element type {element_type} is not valid.")
        return node_id, link_id
    
    def get_Cin(self, id, pollutant, element_type):
        """
        Get the current concentration of a pollutant at a node/link.
        """

        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        if element_type == ElementType.Links:
            id = self.link_to_node[id]

        Cin = self.storage._get_state(element_id=id, pollutant=pollutant)

        # if state is initialized to nan, it's the first in a series of treatment methods; grab from model
        
        if np.isnan(Cin):
            if element_type == ElementType.Nodes:
                # Get SWMM parameter
                Cin = self.sim._model.getNodePollut(id, tka.NodePollut.inflowQual.value)[pollutant_index]
            else:
                # Get SWMM parameter
                Cin = self.sim._model.getLinkPollut(id, tka.LinkPollut.reactorQual.value)[pollutant_index]

        return Cin
    
    def set_Cin(self, value, id, pollutant, element_type):
        """
        Set the current concentration of a pollutant at a node/link.
        """

        # always set the model and state; the model value only ends up getting used if it's the last in a series of treatment methoods
        if element_type == ElementType.Links:
            raise ValueError("set_Cin does not work for links.")
            #self.sim._model.setLinkPollut(id, pollutant, value)
            ##node_id = self._link_to_node_id(id)
            #self.storage._set_state(element_id=id, pollutant=pollutant, value=value)
        else:

            # IMPORTANT here we get the pollut concentration twice
            # once from the model and once from the storage
            # since the model can't be updated multiple times within one timestep, we store it in 'state'
            # 
            self.sim._model.setNodePollut(id, pollutant, value)
            self.storage._set_state(element_id=id, pollutant=pollutant, value=value)



    #def _link_to_node_id(self, link_id):
    #    if link_id not in self.model.links().index:
    #        raise ValueError(f"Link {link_id} not found in model.")
    #    return self.model.links().InletNode.to_dict()[link_id]

    def updateWQState(self):
        """
        Runs the selected water quality method (except CSTR) and updates
        the pollutant concentration during a SWMM simulation.
        """

        if self.sim._advance_seconds:
            raise(PySWMMStepAdvanceNotSupported)

        # for each config, run the water quality method
        for c in self.config:

            # if it's the first time step at the element id, set the state to 0
            # priority is normalised such that 0 is the highest priority and priorities increment by 1

            if c.element_type == ElementType.Subcatchments:
                element_id = self.subcatchment_to_node[c.element_id]
            else:
                element_id = c.element_id

            if c.priority == 1:
                self.storage._set_state(element_id=element_id, pollutant=c.pollutant, value=np.nan)
        
        for c in self.config:
            self.method[c.method](id=c.element_id, pollutant=c.pollutant, parameters=c.parameters, element_type=c.element_type)


        #Update timestep after water quality methods are completed
        self.last_timestep = self.sim.current_time


    def updateWQState_CSTR(self, index):
        """
        Runs the water quality method CSTR only and updates the pollutant
        concentration during a SWMM simulation.
        """

        if self.sim._advance_seconds:
            raise(PySWMMStepAdvanceNotSupported)

        # Parse all the elements and their parameters in the config dictionary
        for element_id, element_info in self.config:
            attribute = self.config[element_id]['method']
            element_type = self.config[element_id]['type']
            if element_type == "node":
                element_type = ElementType.Nodes
            else:
                print("CSTR does not work for links.")
            # Call the water quality method for each element
            self.method[attribute](index, element_id, self.config[element_id]['pollutant'], self.config[element_id]['parameters'], element_type)

        #Update timestep after water quality methods are completed
        self.last_timestep = self.sim.current_time

    def _CreateCache(self):
        """
        Create a cache for water quality related data so that it doesn't have to be recalculated at every simulation step and element. Method dependant, so cache is only generated for specified methods.
        """

        self.cache = {}
        
        # get the unique WaterQuality methods

        methods = np.unique([cfg.method for cfg in self.config])

        if "DryWeatherLoading" in methods:
            DryWeatherLoadingCache = _CreateDryWeatherLoadingCache(self.model)
            self.cache.update(DryWeatherLoadingCache)

        #if "CustomPollutLoading" in methods:
            #CustomPollutLoadingCache = _CreateCustomPollutLoadingCache(self.model, self.config)

        if "CustomPollutProfile" in methods:
            simulation_timestep = self.model.inp.options.loc["WET_STEP",:].values[0]
            profile_timestep_seconds = int(pd.Timedelta(simulation_timestep).total_seconds())
            
            first_instance = np.where([cfg.method == "CustomPollutProfile" for cfg in self.config])[0][0]
            custom_profile = load_custom_daily_profile(
                self.config[first_instance].parameters["filename"],
                element_type=ElementType.Subcatchments,
                model=self.model)
            
                    # Get the profile timestep in hours
            profile_timestep = (custom_profile.index[1].hour*60+custom_profile.index[1].minute - custom_profile.index[0].hour*60+custom_profile.index[0].minute) * 60


            # Resample the custom profile to match the profile timestep in seconds
            #custom_profile = custom_profile.resample(f'{profile_timestep_seconds}S').interpolate(method='linear')


            self.cache.update({"custom_profile": custom_profile, "profile_timestep": profile_timestep})

    def _ConstantWasteLoad(self, id, pollutant, parameters, element_type):

        """
        Since the PySWMM API can only edit the pollutant concentration at nodes/links, not the DWF concentration, we need to calculate the new concentration based on the DWF load. 

        This is done by calculating the existing load [mg], adding the new load [mg], and dividing by the new flow [m^3/s] to get the new concentration [mg/L].
        """

        _standardize_parameters(parameters, method="ConstantWasteLoad")


        # Get current time
        current_step = self.sim.current_time
        # Calculate model dt in seconds
        dt = (current_step - self.last_timestep).total_seconds()



        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)


        Cin = self.get_Cin(id, pollutant, element_type)
        
        if element_type == ElementType.Nodes:
            # Get SWMM parameter
            Q = self.sim._model.getNodeResult(id, tka.NodeResults.totalinflow.value)
            # Set new concentration
        else:
            # Get SWMM parameter
            Q = self.sim._model.getLinkResult(id, tka.LinkResults.newFlow.value)


        V = Q * dt
        Lin = Cin * V
        Lnew = Lin + parameters["L"]

        if V != 0:
            Cnew = Lnew / V
        else: 
            Cnew = Cin

        self.set_Cin(value=Cnew, id=id, pollutant=pollutant, element_type=element_type)

    def _DryWeatherLoading(self, id, pollutant, parameters, element_type):

        """
        Apply dry-weather loading, permitting the user to set spatially-heterogenous pollutant loading rates.

        Since the PySWMM API doesn't allow for the direct editing of DWF inflows or concentrations, we need to lookup the DWF loading from the model inp file and apply them at the nodes
        """

        _standardize_parameters(parameters, method="DryWeatherLoading")


        # Get current time
        current_step = self.sim.current_time
        # Calculate model dt in seconds
        dt = (current_step - self.last_timestep).total_seconds()
        
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        if element_type == ElementType.Nodes:
            # Get SWMM parameter
            Cin = self.sim._model.getNodePollut(id, tka.NodePollut.inflowQual.value)[pollutant_index]
            Q = self.sim._model.getNodeResult(id, tka.NodeResults.totalinflow.value)
            # Set new concentration
        else:
            # Get SWMM parameter
            Cin = self.sim._model.getLinkPollut(id, tka.LinkPollut.reactorQual.value)[pollutant_index]
            Q = self.sim._model.getLinkResult(id, tka.LinkResults.newFlow.value)

        Ldwf = get_dwf_load(element=id, pollutant=pollutant, datestamp=current_step, dt=dt, cache=self.cache)

        V = Q * dt
        Lin = Cin * V
        Lnew = Lin + - Ldwf + Ldwf * parameters["multiplier"]

        if V != 0:
            Cnew = Lnew / V
        else: 
            Cnew = Cin

        self.set_Cin(value=Cnew, id=id, pollutant=pollutant, element_type=element_type)


    def _CustomPollutProfile(self, id, pollutant, parameters, element_type):
        
        #_standardize_parameters(parameters, method="CustomPollutProfile")
        
        # if applied to a subcatchment, subcatchment must have an outlet node (as opposed to another subcatchment)
        #if element_type == ElementType.Subcatchments:
        #outlets = self.model.subcatchments().Outlet.to_dict()
        #else:
        #    raise ValueError(f"CustomPollutProfile method can only be applied to subcatchments. Element type {element_type} is not valid.")
        
        
        #nodes = self.model.nodes().index
        node_id = self.subcatchment_to_node[id]


        #if outlets[id] not in nodes:
        #    raise ValueError(f"Subcatchment {id} does not have an outlet node in the model.")
        
        profile = self.cache["custom_profile"]
        profile_timestep = self.cache["profile_timestep"]
        #profile_timestep = 100
        current_step = self.sim.current_time
        dt = (current_step - self.last_timestep).total_seconds()
        #dt = 100
        # get the loading rate from the profile
        loading = profile.loc[profile.index.asof(current_step.time()), id]
        #loading = 10
        # convert to [unit]/s
        loading_rate = loading / profile_timestep # s^-1

        # apply to model timestep
        timestep_loading = loading_rate * dt

        # get flow rate and volume        
        # get concentration in
        Cin = self.get_Cin(node_id, pollutant, ElementType.Nodes)


        Q = self.sim._model.getNodeResult(node_id, tka.NodeResults.totalinflow.value)
        #Q = 10
        V = Q * dt

        # add to Cin as concentration
        if V != 0:
            Cin += timestep_loading / V
        else:
            Cin = 0

        # set new concentration
        self.set_Cin(value=Cin, id=node_id, pollutant=pollutant, element_type=element_type)
        
    def _EventMeanConc(self, id, pollutant, parameters, element_type):
        """
        Event Mean Concentration Treatment (SWMM Water Quality Manual, 2016)
        Treatment results in a constant concentration.

        Treatment method parameters required:
        C = constant treatment concentration for each pollutant (SI/US: mg/L)
        """
    
        _standardize_parameters(parameters, method="EventMeanConc")

        if element_type == ElementType.Nodes:
            # Set new concentration
            self.sim._model.setNodePollut(id, pollutant, parameters["C"])
        else:
            # Set new concentration
            self.sim._model.setLinkPollut(id, pollutant, parameters["C"])


    def _ConstantRemoval(self, id, pollutant, parameters, element_type):
        """
        CONSTANT REMOVAL TREATMENT (SWMM Water Quality Manual, 2016)
        Treatment results in a constant percent removal.

        R = pollutant removal fraction (unitless)
        """
        # Get pollutant index
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        if element_type == ElementType.Nodes:
            # Get SWMM parameter
            Cin = self.sim._model.getNodePollut(id, tka.NodePollut.inflowQual.value)[pollutant_index]
            # Calculate new concentration
            Cnew = (1-parameters["R"])*Cin
            # Set new concentration
        else:
            # Get SWMM parameter
            Cin = self.sim._model.getLinkPollut(id, tka.LinkPollut.reactorQual.value)[pollutant_index]
            # Calculate new concentration
            Cnew = (1-parameters["R"])*Cin
            # Set new concentration


        self.set_Cin(value=Cnew, id=id, pollutant=pollutant, element_type=element_type)

    def _CoRemoval(self, id, pollutant, parameters, element_type):
        """
        CO-REMOVAL TREATMENT (SWMM Water Quality Manual, 2016)
        Removal of some pollutant is proportional to the removal of
        some other pollutant.

        R1 = pollutant removal fraction (unitless)
        R2 = pollutant removal fraction for other pollutant (unitless)
        """

        _standardize_parameters(parameters, method="CoRemoval")

        # Get pollutant index
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        if element_type == ElementType.Nodes:
            # Get SWMM parameter
            Cin = self.sim._model.getNodePollut(id, tka.NodePollut.inflowQual.value)[pollutant_index]
            # Calculate new concentration
            Cnew = (1-parameters["R1"]*parameters["R2"])*Cin
            # Set new concentration
        else:
            # Get SWMM parameter
            Cin = self.sim._model.getLinkPollut(id, tka.LinkPollut.reactorQual.value)[pollutant_index]
            # Calculate new concentration
            Cnew = (1-parameters["R1"]*parameters["R2"])*Cin
            # Set new concentration


        self.set_Cin(value=Cnew, id=id, pollutant=pollutant, element_type=element_type)


    def _ConcDependRemoval(self, id, pollutant, parameters, element_type):
        """
        CONCENTRATION-DEPENDENT REMOVAL (SWMM Water Quality Manual, 2016)
        When higher pollutant removal efficiencies occur with higher
        influent concentrations.

        R_l = lower removal rate (unitless)
        BC  = boundary concentration that determines removal rate (SI/US: mg/L)
        R_u = upper removal rate (unitless)
        """

        _standardize_parameters(parameters, method="ConcDependRemoval")

        # Get pollutant index
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        if element_type == ElementType.Nodes:
            # Get SWMM parameter
            Cin = self.sim._model.getNodePollut(id, tka.NodePollut.inflowQual.value)[pollutant_index]
            # Calculate removal
            R = (1-np.heaviside((Cin-parameters["BC"]), 0))\
            *parameters["R_l"]+np.heaviside((Cin\
            -parameters["BC"]),0)*parameters["R_u"]
            # Calculate new concentration
            Cnew = (1-R)*Cin
            # Set new concentration
        else:
            # Get SWMM parameter
            Cin = self.sim._model.getLinkPollut(id, tka.LinkPollut.reactorQual.value)[pollutant_index]
            # Calculate removal
            R = (1-np.heaviside((Cin-parameters["BC"]), 0))\
            *parameters["R_l"]+np.heaviside((Cin\
            -parameters["BC"]),0)*parameters["R_u"]
            # Calculate new concentration
            Cnew = (1-R)*Cin
            # Set new concentration

        self.set_Cin(value=Cnew, id=id, pollutant=pollutant, element_type=element_type)

    def _NthOrderReaction(self, id, pollutant, parameters, element_type):
        """
        NTH ORDER REACTION KINETICS (SWMM Water Quality Manual, 2016)
        When treatment of pollutant X exhibits n-th order reaction kinetics
        where the instantaneous reaction rate is kC^n.

        k   = reaction rate constant (SI: m/hr, US: ft/hr)
        n   = reaction order (first order, second order, etc.) (unitless)
        """

        _standardize_parameters(parameters, method="NthOrderReaction")

        # Get pollutant index
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        # Get current time
        current_step = self.sim.current_time
        # Calculate model dt in seconds
        dt = (current_step - self.last_timestep).total_seconds()

        if element_type == ElementType.Nodes:
            # Get SWMM parameter
            C = self.sim._model.getNodePollut(id, tka.NodePollut.reactorQual.value)[pollutant_index]
            # Calculate treatment
            Cnew = C - (parameters["k"]*(C**parameters["n"])*dt)
            # Set new concentration
        else:
            # Get SWMM parameter
            C = self.sim._model.getLinkPollut(id, tka.LinkPollut.reactorQual.value)[pollutant_index]
            # Calculate treatment
            Cnew = C - (parameters["k"]*(C**parameters["n"])*dt)
            # Set new concentration

        self.set_Cin(value=Cnew, id=id, pollutant=pollutant, element_type=element_type)

    def _kCModel(self, id, pollutant, parameters, element_type):
        """
        K-C_STAR MODEL (SWMM Water Quality Manual, 2016)
        The first-order model with background concentration made popular by
        Kadlec and Knight (1996) for long-term treatment performance of wetlands.

        k   = reaction rate constant (SI: m/hr, US: ft/hr)
        C_s = constant residual concentration that always remains (SI/US: mg/L)
        """
        
        _standardize_parameters(parameters, method="kCModel")

        # Get pollutant index
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        if element_type == ElementType.Nodes:
            # Get SWMM parameters
            Cin = self.sim._model.getNodePollut(id, tka.NodePollut.inflowQual.value)[pollutant_index]
            d = self.sim._model.getNodeResult(id, tka.NodeResults.newDepth.value)
            hrt = self.sim._model.getNodeResult(id, tka.NodeResults.hyd_res_time.value)
            # Calculate removal
            if d != 0.0 and Cin != 0.0:
                R = np.heaviside((Cin-parameters["C_s"]), 0)\
                *((1-np.exp(-parameters["k"]*hrt/d))*(1-parameters["C_s"]/Cin))
            else:
                R = 0
            # Calculate new concentration
            Cnew = (1-R)*Cin
            # Set new concentration
            self.sim._model.setNodePollut(id, pollutant, Cnew)
            self.storage._set_state(element_id=id, pollutant=pollutant, value=Cnew)
        else:
            print("kCModel does not work for links.")


    def _GravitySettling(self, id, pollutant, parameters, element_type):
        """
        GRAVITY SETTLING (SWMM Water Quality Manual, 2016)
        During a quiescent period of time within a storage volume, a fraction
        of suspended particles will settle out.

        k   = reaction rate constant (SI: m/hr, US: ft/hr)
        C_s = constant residual concentration that always remains (SI/US: mg/L)
        """

        _standardize_parameters(parameters, method="GravitySettling")

        # Get pollutant index
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        # Get current time
        current_step = self.sim.current_time
        # Calculate model dt in seconds
        dt = (current_step - self.last_timestep).total_seconds()

        if element_type == ElementType.Nodes:
            # Get SWMM parameters
            Cin = self.sim._model.getNodePollut(id, tka.NodePollut.inflowQual.value)[pollutant_index]
            Qin = self.sim._model.getNodeResult(id, tka.NodeResults.totalinflow.value)
            d = self.sim._model.getNodeResult(id, tka.NodeResults.newDepth.value)
            if d != 0.0:
                # Calculate new concentration
                Cnew = np.heaviside((0.1-Qin), 0)*(parameters["C_s"]\
                +(Cin-parameters["C_s"])*np.exp(-parameters["k"]/d*dt/3600))\
                +(1-np.heaviside((0.1-Qin), 0))*Cin
            else:
                Cnew = np.heaviside((0.1-Qin), 0)*parameters["C_s"]\
                +(Cin-parameters["C_s"])+(1-np.heaviside((0.1-Qin), 0))*Cin
            # Set new concentration
        else:
            # Get SWMM parameters
            C = self.sim._model.getLinkPollut(id, tka.LinkPollut.reactorQual.value)[pollutant_index]
            Q = self.sim._model.getLinkResult(id, tka.LinkResults.newFlow.value)
            d = self.sim._model.getLinkResult(id, tka.LinkResults.newDepth.value)
            if d != 0.0:
                # Calculate new concentration
                Cnew = np.heaviside((0.1-Q), 0)*(parameters["C_s"]\
                +(C-parameters["C_s"])*np.exp(-parameters["k"]/d*dt/3600))\
                +(1-np.heaviside((0.1-Q), 0))*C
            else:
                Cnew = np.heaviside((0.1-Q), 0)*parameters["C_s"]\
                +(C-parameters["C_s"])+(1-np.heaviside((0.1-Q), 0))*C
            # Set new concentration

        self.set_Cin(value=Cnew, id=id, pollutant=pollutant, element_type=element_type)

    def _SewageFlux(self, id, pollutant, parameters, element_type):
        """
        Sewage settling and resuspension model (Lederberger et al, 2019)
        """

        #if element_type == ElementType.Nodes:
        #    raise NotImplementedError("SewageSettling does not work for nodes.")

        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)


        
        node_id, link_id = self.split_ids(id, element_type)

        Cin = self.get_Cin(node_id, pollutant, ElementType.Nodes)

        d = self.sim._model.getLinkResult(link_id, tka.LinkResults.newDepth.value)
        V = self.sim._model.getLinkResult(link_id, tka.LinkResults.newVolume.value)

        Qin = self.sim._model.getLinkResult(link_id, tka.LinkResults.newFlow.value) # [m^3/s]


        
        Sin = self.storage._get_storage(element_id=node_id, pollutant=pollutant)
        

        current_step = self.sim.current_time
        dt = (current_step - self.last_timestep).total_seconds() #[seconds]

        # NOTE: I'm replacing A/V(t) with 1/d(t)
        # in lerderberger, a rectangular linear reservoir is used, which has a constant A
        # when calculating settling in conduits, you should technically use A(t)/V(t), since the surface of the area changes with depth
        # also, you should use an effective depth, since the depth near the walls of the pipe will be lower
        # in the absence of a better method, I'm using 1/d(t) as a proxy for A(t)/V(t), which is the depth to invert
        
        Msewer = 0
        
        if V == 0:
            Fsett = 0
        else:
            Msewer = Cin * V # [mg/L * m^3 = mg]
            Fsett = Msewer * (1/d) * parameters["v_sett"] # [mg/s]

        # if the settling mass is greater than the total mass, then the settling mass is set to the total mass
        Msett = Fsett * dt # [mg]
        if Msett > Msewer:
            Msett = Msewer


        # Sin is equal to 'Particle mass settled in the sediment compartment' (Lederberger et al, 2019)
        rresus = parameters["Resus_max"] * (Qin**parameters["n"])/(Qin**parameters["n"] + parameters["Qhalf"]**parameters["n"]) # [1/s] 
        Fresus = Sin * rresus # [mg/s]
        Mresus = Fresus * dt # [mg]


        if Mresus > Sin:
            Mresus = Sin

        # negative Mnet means settling > resuspension
        Mnet = Mresus - Msett # [mg]
        
        if Mnet < 0: # settling
            #if -Mnet > Sin:
            #    Meff = -Sin

            if (Sin - Mnet) >= parameters["Smax"]:
                remaining_storage = parameters["Smax"] - Sin
                Meff = -remaining_storage
            else:
                Meff = Mnet

        elif Mnet > 0: # resuspension
            if Mnet > Msewer:
                Meff = Msewer
            else:
                Meff = Mnet
        else:
            Meff = 0

        Snew = Sin - Meff

        if V != 0:
            Cnew = (Msewer + Meff)/V
        else:
            Cnew = Cin


        """
        # adjust the inflow concentration
        # if no flow, make no changes
        if V != 0:
            Cnew = Mnet/V + Cin
            Snew = Sin - Meff

        else:
            Cnew = Cin
            Snew = Sin
        
        if Cnew < 0:
            Cnew = 0

        # if the maximum storage is exceeded, the difference is added to the outflow concentration
        if Snew > parameters["Smax"]:
            Cnew =+ (Snew - parameters["Smax"])/V
            Snew = parameters["Smax"]
                    
        self.storage._set_storage(element_id=id,pollutant=pollutant,value=Snew)
        
        # positive flux - resuspension
        # negative flux - settling
        #flux = Cin - Cnew
        self.storage._set_flux(element_id=id,pollutant=pollutant,value=(Cnew - Cin)*V)
        """

        self.storage._set_flux(element_id=node_id,pollutant=pollutant,value=(Cnew - Cin)*V)
        self.storage._set_storage(element_id=node_id,pollutant=pollutant,value=Snew)
        self.set_Cin(value=Cnew, id=node_id, pollutant=pollutant, element_type=element_type)

    def _ViralDecay(self, id, pollutant, parameters, element_type):
        """
        Sewage settling and resuspension model (Lederberger et al, 2019)
        """


        # get the depth and volume of the conduit
        
        node_id, link_id = self.split_ids(id, element_type)

        d = self.sim._model.getLinkResult(link_id, tka.LinkResults.newDepth.value)
        V = self.sim._model.getLinkResult(link_id, tka.LinkResults.newVolume.value)

        # Get pollutant index
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        Cin = self.get_Cin(node_id, pollutant, element_type)

        # Get current time
        current_step = self.sim.current_time
        # Calculate model dt in seconds
        dt = (current_step - self.last_timestep).total_seconds()

        # get the copollutant concentration (this is grapped prior to the settling/resuspension, since the API seems to update things at each timestep)
        copollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, parameters["copollutant"])
        CPin = self.sim._model.getLinkPollut(link_id, tka.LinkPollut.reactorQual.value)[copollutant_index] # [mg/L]

        # get the copollutant flux (net sett/resus [mg])
        CPflux = self.storage._get_flux(element_id=node_id, pollutant=parameters["copollutant"])
    

        if V != 0:
            CPnew = CPin + CPflux/V
        else:
            CPnew = CPin
            
        if CPin != 0:
            CPchange = (CPnew - CPin)/CPin
        else:
            CPchange = 0

        # get the copollutant (tss) storage [mg]
        CP_Snew = self.storage._get_storage(element_id=node_id, pollutant=parameters["copollutant"])


        # get the viral flux [mg * copies/mg]
        Mflux = CPflux * Cin # [copies]

        # get the viral load in storage [copies]
        Sin = self.storage._get_storage(element_id=node_id, pollutant=pollutant)
        
        # calculate the new viral load in storage [copies]
        Snew = Sin - Cin * CPchange

        # calculate the viral concentration in storage [copies/mg]
        if CP_Snew != 0:
            Sconc = Snew / CP_Snew
        else:
            Sconc = 0

        # apply decay to storage [copies/mg]
        Sdecay = Sconc - parameters["k"] * (Sconc ** parameters["n"]) * dt
        #Sdecay = Snew


        Sout = Sdecay * CP_Snew
        #
        if Sdecay < 0:
            Sdecay = 0

        # set the new storage (after applying decay)
        self.storage._set_storage(element_id=id,pollutant=pollutant,value=Sout)

        # get the viral conc in
        Cnew = Cin + CPchange * Cin

        # apply decay to viral concentration out [copies/mg]
        Cdecay = Cnew - (parameters["k"]*(Cnew**(parameters["n"])*dt))
        
        if Cdecay < 0:
            Cdecay = 0
            warnings.warn("Viral concentration has decayed to zero. Consider increasing the decay rate or changing the decay model.")

        self.set_Cin(value=Cdecay, id=node_id, pollutant=pollutant, element_type=element_type)

    """
    Need to add conduit velocity getter to swmm/pyswmm
    def _Erosion(self, id, pollutant, parameters, flag):

        ENGELUND-HANSEN EROSION (1967)
        Engelund and Hansen (1967) developed a procedure for predicting stage-
        discharge relationships and sediment transport in alluvial streams.

        w   = channel width (SI: m, US: ft)
        So  = bottom slope (SI: m/m, US: ft/ft)
        Ss  = specific gravity of sediment (for soil usually between 2.65-2.80)
        d50 = mean sediment particle diameter (SI/US: mm)
        d   = depth (SI: m, US: ft)
        qt  = sediment discharge per unit width (SI: kg/m-s, US: lb/ft-s)
        Qt  = sediment discharge (SI: kg/s, US: lb/s)


        parameters = parameters

        # Get current time
        current_step = self.sim.current_time
        # Calculate model dt in seconds
        dt = (current_step - self.last_timestep).total_seconds()
        # Updating reference step
        self.last_timestep = current_step

        if self.flag == 0:
            print("Erosion does not work for nodes.")
        else:
            # Get SWMM parameters
            Cin = self.sim._model.getLinkC2(id, pollutant)
            Q = self.sim._model.getLinkResult(id, 0)
            d = self.sim._model.getLinkResult(id, 1)
            v = self.sim._model.getConduitVelocity(id)

            # Erosion calculations for US units
            if self.sim._model.getSimUnit(0) == "US":
                g = 32.2            # ft/s^2
                ρw = 62.4           # lb/ft^3
                mm_ft = 0.00328     # ft/mm
                lb_mg = 453592      # mg/lb
                L_ft3 = 0.0353      # ft3/L
                if v != 0.0:
                    qt = 0.1*(1/((2*g*parameters["So"]*d)/v**2))*((d\
                    *parameters["So"]/((parameters["Ss"]-1)*parameters["d50"]))\
                    *(1/mm_ft))**(5/2)*parameters["Ss"]*ρw*((parameters["Ss"]-1)\
                    *g*(parameters["d50"]*mm_ft)**3)**(1/2) # lb/ft-s
                    Qt = parameters["w"]*qt       # lb/s
                else:
                    Qt = 0.0
                if Q !=0.0:
                    Cnew = (Qt/Q)*lb_mg*L_ft3   # mg/L
                    Cnew = max(Cin, Cin+Cnew)
                    # Set new concentration
                    self.sim._model.setLinkPollut(id, pollutant, Cnew)

            # Erosion calculations for SI units
            else:
                g = 9.81            # m/s^2
                ρw = 1000           # kg/m^3
                mm_m = 0.001        # m/mm
                kg_mg = 1000000     # mg/kg
                L_m3 =  0.001       # m3/L
                if v != 0.0:
                    qt = 0.1*(1/((2*g*parameters["So"]*d)/v**2))*((d\
                    *parameters["So"]/((parameters["Ss"]-1)*parameters["d50"]))\
                    *(1/mm_m))**(5/2)*parameters["Ss"]*ρw*((parameters["Ss"]-1)\
                    *g*(parameters["d50"]*mm_m)**3)**(1/2) # kg/m-s
                    Qt = parameters["w"]*qt       # kg/s
                else:
                    Qt = 0.0
                if Q != 0.0:
                    Cnew = (Qt/Q)*L_m3*kg_mg    # mg/L
                    Cnew = max(Cin, Cin+Cnew)
                    # Set new concentration
                    self.sim._model.setLinkPollut(id, pollutant, Cnew)
    """


    def _CSTR_tank(self, t, C, Qin, Cin, Qout, V, k, n):
        """
        UNSTEADY CONTINUOUSLY STIRRED TANK REACTOR (CSTR)
        CSTR is a common model for a chemical reactor. The behavior of a CSTR
        is modeled assuming it is not in steady state. This is because
        outflow, inflow, volume, and concentration are constantly changing.

        NOTE: You do not need to call this method, only the CSTR_solver.
        CSTR_tank is intitalized in __init__ in Node_Treatment.
        """
        dCdt = (Qin*Cin - Qout*C)/V + k*C**n
        return dCdt


    def _CSTRSolver(self, index, id, pollutant, parameters, element_type):
        """
        UNSTEADY CONTINUOUSLY STIRRED TANK REACTOR (CSTR) SOLVER
        CSTR is a common model for a chemical reactor. The behavior of a CSTR
        is modeled assuming it is not in steady state. This is because
        outflow, inflow, volume, and concentration are constantly changing.
        Therefore, Scipy.Integrate.ode solver is used to solve for concentration.

        NOTE: You only need to call this method, not CSTR_tank. CSTR_tank is
        intitalized in __init__ in Node_Treatment.

        k   = reaction rate constant (SI/US: 1/s)
        n   = reaction order (first order, second order, etc.) (unitless)
        c0  = intital concentration inside reactor (SI/US: mg/L)
        """

        # Get current time
        current_step = self.sim.current_time
        # Calculate model dt in seconds
        dt = (current_step - self.last_timestep).total_seconds()
        # Get pollutant index
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        if element_type == ElementType.Nodes:
            # Get SWMM parameters
            Cin = self.sim._model.getNodePollut(id, tka.NodePollut.inflowQual.value)[pollutant_index]
            Qin = self.sim._model.getNodeResult(id, tka.NodeResults.totalinflow.value)
            Qout = self.sim._model.getNodeResult(id, tka.NodeResults.outflow.value)
            V = self.sim._model.getNodeResult(id, tka.NodeResults.newVolume.value)

            # Parameterize solver
            self.solver.set_f_params(Qin, Cin, Qout, V, parameters["k"], parameters["n"])
            # Solve ODE
            if index == 0:
                self.solver.set_initial_value(parameters["c0"], 0.0)
                self.solver.integrate(self.solver.t+dt)
            else:
                self.solver.set_initial_value(self.solver.y, self.solver.t)
                self.solver.integrate(self.solver.t+dt)
            # Set new concentration
            self.sim._model.setNodePollut(id, pollutant, self.solver.y[0])
        else:
            print("CSTR does not work for links.")


    def _Phosphorus(self, index, id, pollutant, parameters, element_type):
        """
        LI & DAVIS BIORETENTION CELL TOTAL PHOSPHOURS MODEL (2016)
        Li and Davis (2016) developed a dissolved and particulate phosphorus
        model for bioretention cells.

        B1    = coefficient related to the rate at which Ceq approaches Co (SI/US: 1/s)
        Ceq0  = initial DP or PP equilibrium concentration value for an event (SI/US: mg/L)
        k     = reaction rate constant (SI/US: 1/s)
        L     = depth of soil media (length of pathway) (SI: m, US: ft)
        A     = cross-sectional area (SI: m^2, US: ft^2)
        E     = filter bed porosity (unitless)
        """

        parameters = parameters
        t = 0
        # Get pollutant index
        pollutant_index = self.sim._model.getObjectIDIndex(tka.ObjectType.POLLUT, pollutant)

        if element_type == ElementType.Nodes:
            # Get SWMM parameters
            Cin = self.sim._model.getNodePollut(id, tka.NodePollut.inflowQual.value)[pollutant_index]
            Qin = self.sim._model.getNodeResult(id, tka.NodeResults.totalinflow.value)
            # Time calculations for phosphorus model
            if Qin >= 0.01:
                # Get current time
                current_step = self.sim.current_time
                # Calculate model dt in seconds
                dt = (current_step - self.last_timestep).total_seconds()
                # Accumulate time elapsed since water entered node
                t = t + dt
                # Updating reference step
                self.last_timestep = current_step
                # Calculate new concentration
                Cnew = (Cin*np.exp((-parameters["k"]*parameters["L"]\
                    *parameters["A"]*parameters["E"])/Qin))+(parameters["Ceq0"]\
                    *np.exp(parameters["B1"]*t))*(1-(np.exp((-parameters["k"]\
                    *parameters["L"]*parameters["A"]*parameters["E"])/Qin)))
                # Set new concentration
                self.sim._model.setNodePollut(id, pollutant, Cnew)
            else:
                t = 0
        else:
            print("Phosphorus does not work for links.")