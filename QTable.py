import numpy as np
import xarray as xr


class QTable:

    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2

        size = (len(agent1.possible_states), len(agent2.possible_states), len(agent1.possible_actions),
                len(agent2.possible_actions))

        np_table = np.random.uniform(0, 10, size)

        self.table = xr.DataArray(np_table,
                                  coords=[agent1.possible_states, agent2.possible_states, agent1.possible_actions,
                                          agent2.possible_actions],
                                  dims=[agent1.name + "_state", agent2.name + "_state", agent1.name + "_action",
                                        agent2.name + "_action"])


    def get_table(self):
        return self.table.sel(**{self.agent1.name + "_state": self.agent1.current_state,
                                 self.agent2.name + "_state": self.agent2.current_state}, drop=True)
