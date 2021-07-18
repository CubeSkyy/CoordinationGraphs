import json
from itertools import product

import numpy as np
import xarray as xr
from Agent import Agent
from BestResponse import BestResponse
from PayoutFunction import PayoutFunction
from QTable import QTable

with open("grid_4/coordination_graph.json") as f:
    data = json.load(f)

agents = {}

# Agent initialization
for agent_name in data["agents"]:
    agents[agent_name] = Agent(agent_name)

# Init Qtables
qTables = {}
all_payouts = [] #For debug only

for link in data["connections"]:
    agent1 = agents[link[0]]
    agent2 = agents[link[1]]
    qTable = QTable(agent1, agent2)
    payoutFunction = PayoutFunction([agent1, agent2], qTable.get_table().data)
    all_payouts.append(payoutFunction) #For debug only
    agent1.q_tables[agent2.name] = qTable
    agent1.payout_functions[agent2.name] = payoutFunction
    agent2.q_tables[agent1.name] = qTable
    agent2.payout_functions[agent1.name] = payoutFunction

#TODO Only pass correct arguments to set_value (instead of doing an iteration everyime)
#TODO Combine new_function and best_response?
#TODO Do sanity check by iterating over all possible joint actions and print payoff
#TODO Remove name indexing ("a1_0", etc)

def print_all_payouts(agents):
    max = ("", -1)
    action_product = list(product(*[[x + "_" + y for y in agents[x].possible_actions] for x in agents]))
    for joint_action in action_product:
        _sum = 0
        for payout in all_payouts:
            _sum += payout.get_value(joint_action)
        if _sum > max[1]:
            max = joint_action, _sum
        print(str(joint_action) + " " + str(_sum))
    print("\nBest joint action: ", max)

def variable_elimination(agents, order=None):
    if order is not None:
        elimination_agents = [agents[agent_name] for agent_name in order]
    else:
        elimination_agents = agents.values()

    # First Pass
    for agent in elimination_agents:

        #For every agent that depends on current agent
        arguments = [dependant for dependant in agent.payout_functions.keys() if dependant != agent.name]
        action_product = list(
             product(*[[x + "_" + y for y in agents[x].possible_actions] for x in arguments]))

        new_function = PayoutFunction([agents[agent_name] for agent_name in arguments])
        best_response = BestResponse([agents[agent_name] for agent_name in arguments])

        #For every action pair of dependant agents
        for joint_action in action_product:
            _max = ("-1", -1)
            #Figure out the max and maxArg of current agent actions
            for agent_action in agent.possible_actions:
                _sum = 0
                #Maximizing the sum of every local payout function
                for function in list(agent.payout_functions.values()):
                    _sum += function.get_value((agent.name + "_" + agent_action, *joint_action))
                if _sum >= _max[1]:
                    _max = (agent_action ,_sum)

            #Save new payout and best response
            best_response.set_value(joint_action, _max[0])
            new_function.set_value(joint_action, _max[1])
        agent.best_response = best_response

        #Delete the old and add the new payout functions from dependants
        for agent_name in arguments:
            del agents[agent_name].payout_functions[agent.name]
            dependent_agent = [agent for agent in arguments if agent != agent_name]
            if len(dependent_agent) > 0:
                agents[agent_name].payout_functions[dependent_agent[0]] = new_function
        if len(arguments) == 1:
            name = list(arguments)[0]
            agents[name].payout_functions[name] = new_function

    #Second Pass, Reverse Order, excluding the last agent
    last_agent = list(elimination_agents)[-1]
    actions = {last_agent.name : last_agent.best_response.table.data[()]}
    for agent in list(elimination_agents)[-2::-1]:
        actions[agent.name] = agent.best_response.get_value(actions)

    print("Variable Elimination Result:")
    for key, value in sorted(actions.items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value), end=', ')

print_all_payouts(agents)
print("\n")
variable_elimination(agents)
