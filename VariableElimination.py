import json
from itertools import product
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
    agent1.payout_functions.append(payoutFunction)
    agent1.dependant_agents.add(agent2.name)
    agent2.q_tables[agent1.name] = qTable
    agent2.payout_functions.append(payoutFunction)
    agent2.dependant_agents.add(agent1.name)


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
    for agent in elimination_agents[:-1]:

        #For every agent that depends on current agent
        # dependant_agents = [dependant for dependant in agent.payout_functions.keys() if dependant != agent.name]
        dependant_agents = agent.dependant_agents
        action_product = list(
             product(*[[x + "_" + y for y in agents[x].possible_actions] for x in dependant_agents]))

        new_function = PayoutFunction([agents[agent_name] for agent_name in dependant_agents])
        best_response = BestResponse([agents[agent_name] for agent_name in dependant_agents])

        #For every action pair of dependant agents
        for joint_action in action_product:
            _max = ("-1", -1)
            #Figure out the max and maxArg of current agent actions
            for agent_action in agent.possible_actions:
                _sum = 0
                #Maximizing the sum of every local payout function
                for function in agent.payout_functions:
                    _sum += function.get_value((agent.name + "_" + agent_action, *joint_action))
                if _sum >= _max[1]:
                    _max = (agent_action ,_sum)

            #Save new payout and best response
            best_response.set_value(joint_action, _max[0])
            new_function.set_value(joint_action, _max[1])
        agent.best_response = best_response
        #Delete the old and add the new payout functions from dependants
        for agent_name in dependant_agents:
            agents[agent_name].payout_functions = [function for function in agents[agent_name].payout_functions if agent.name not in function.agent_names]
            agents[agent_name].dependant_agents.remove(agent.name)

            agents[agent_name].payout_functions.append(new_function)
            agents[agent_name].dependant_agents.update([agent for agent in dependant_agents if agent != agent_name])

    #Second Pass, Reverse Order, excluding the last agent
    last_agent = list(elimination_agents)[-1]
    actions = {last_agent.name : str(last_agent.payout_functions[0].table.argmax().data[()])}
    for agent in list(elimination_agents)[-2::-1]:
        actions[agent.name] = agent.best_response.get_value(actions)

    print("Variable Elimination Result:")
    for key, value in sorted(actions.items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value), end=', ')


print_all_payouts(agents)
print("\n")
variable_elimination(agents, order=["a1", "a2", "a3", "a4"])
