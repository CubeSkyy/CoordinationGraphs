import numpy as np
import pandas as pd
import xarray as xr


class Agent:

    def __init__(self, name):
        self.name = name
        self.num_actions = 2

        self.num_phases = 2
        self.states = ["delay"]
        self.has_period = False
        self.connected_agents = []
        self.possible_states = []
        self.possible_actions = ["0", "1"]

    def build_states(self):
        for state in self.states:
            for i in range(self.num_phases):
                self.possible_states.append(self.name + "_" + state + "_p" + str(i))

        for agent in self.connected_agents:
            for state in agent.states:
                for i in range(agent.num_phases):
                    self.possible_states.append(agent.name + "_" + state + "_p" + str(i))

    def init_QTable(self, randomInit=True):
        if randomInit:
            rng = np.random.default_rng(seed=42)
            self.q_table = pd.DataFrame(
                rng.integers(0, 10, size=(len(self.possible_states), len(self.possible_actions))),
                index=self.possible_states, columns=self.possible_actions, dtype="float64")
        else:
            self.q_table = pd.DataFrame(index=self.possible_states, columns=self.possible_actions,
                                        dtype="float64").fillna(0)

    def init_table(self, randomInit=True):
        actions = [self.possible_actions]
        size = [len(self.possible_actions)]
        agent_names = [self.name]
        for agent in self.connected_agents:
            size.append(len(agent.possible_actions))
            actions.append(agent.possible_actions)
            agent_names.append(agent.name)
        if randomInit:
            np.random.seed(42)
            np_table = np.random.uniform(0, 10, size)
        else:
            np_table = np.zeros(tuple(size))
        agent_names.sort()
        self.table = xr.DataArray(np_table, coords=actions, dims=agent_names)
        # str( self.table .sel(a1="1").idxmax(dim="a2").sel(a3="0").values)

    def init_variables(self):
        self.build_states()
        self.init_QTable()
        self.init_table()
