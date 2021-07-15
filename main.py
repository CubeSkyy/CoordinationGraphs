import json

from Agent import Agent

with open("grid_4/coordination_graph.json") as f:
    graph = json.load(f)

agents = {}

# Agent initialization
for agent_name in graph:
    agents[agent_name] = Agent(agent_name)

# Build coordination graph and initialize variables that depend on graph (State, etc)
for agent_name in graph:
    for child_agent in graph[agent_name]:
        agents[agent_name].connected_agents.append(agents[child_agent])
    agents[agent_name].init_variables()

agents["a1"].table.idxmax(dim="a1")

print("")
