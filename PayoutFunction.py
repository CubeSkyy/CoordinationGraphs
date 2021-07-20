import numpy as np
import xarray as xr


class PayoutFunction:
    def __init__(self, agents, data=None):
        self.agents = agents
        self.agent_names = [agent.name for agent in self.agents]
        if data is None:
            size = tuple([len(agent.possible_actions) for agent in self.agents])
            data = np.zeros(size)

        self.table = xr.DataArray(data,
                                  coords=[agent.possible_actions for agent in self.agents],
                                  dims=[agent.name for agent in self.agents])

    def get_value(self, actions):
        res = self.table
        for action in actions:
            agent_name = action.split("_")[0]
            agent_action = action.split("_")[1]
            if agent_name in self.table.dims:
                res = res.loc[{agent_name: agent_action}]

        return res.data

    def set_value(self, actions, value):
        dims = {}
        for action in actions:
            agent_name = action.split("_")[0]
            agent_action = action.split("_")[1]
            if agent_name in self.table.dims:
                dims[agent_name] = agent_action

        self.table.loc[dims] = value
