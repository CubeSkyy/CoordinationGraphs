import json
from itertools import product
from Agent import Agent
from ActionTable import ActionTable
from QTable import QTable


def init_agents():
    with open("grid_4/coordination_graph.json") as f:
        data = json.load(f)

    agents = {}

    # Agent initialization
    for agent_name in data["agents"]:
        agents[agent_name] = Agent(agent_name)

    for link in data["connections"]:
        agent1 = agents[link[0]]
        agent2 = agents[link[1]]

        qTable = QTable(agent1, agent2)
        payoutFunction = ActionTable([agent1, agent2], qTable.get_table().data)

        agent1.q_tables[agent2.name] = qTable
        agent1.payout_functions.append(payoutFunction)
        agent1.dependant_agents.append(agent2.name)

        agent2.q_tables[agent1.name] = qTable
        agent2.payout_functions.append(payoutFunction)
        agent2.dependant_agents.append(agent1.name)
    return agents


def brute_force(agents):
    max = ("", -1)
    agent_names = [agents[x].name for x in agents]
    action_product = list(product(*[agents[x].possible_actions for x in agents]))
    all_payouts = list(
        dict.fromkeys([item for sublist in [agent.payout_functions for agent in agents.values()] for item in sublist]))
    for joint_action in action_product:
        action_dict = {agent_names[i]: joint_action[i] for i in range(len(agent_names))}
        _sum = 0
        for payout in all_payouts:
            _sum += payout.get_value(action_dict)
        if _sum > max[1]:
            max = action_dict, _sum
    print("\nBest joint action:")
    for key, value in sorted(max[0].items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value), end=', ')
    return max[0]


def variable_elimination(agents, order=None):
    if order is not None:
        elimination_agents = [agents[agent_name] for agent_name in order]
    else:
        elimination_agents = agents.values()

    # First Pass
    for agent in elimination_agents[:-1]:

        # For every agent that depends on current agent
        dependant_agent_names = agent.dependant_agents
        dependant_agents = [agents[agent_name] for agent_name in dependant_agent_names]
        # Create all action possibilities between those agents
        action_product = product(*[agent.possible_actions for agent in dependant_agents])

        new_function = ActionTable(dependant_agents)
        agent.best_response = ActionTable(dependant_agents)

        # For every action pair of dependant agents
        for joint_action in action_product:
            _max = ("-1", -1)
            action_dict = {dependant_agent_names[i]: joint_action[i] for i in range(len(dependant_agent_names))}
            # Figure out the max and maxArg of current agent actions
            for agent_action in agent.possible_actions:
                _sum = 0
                actions = dict({agent.name: agent_action}, **action_dict)
                # Maximizing the sum of every local payout function
                for function in agent.payout_functions:
                    _sum += function.get_value(actions)
                if _sum >= _max[1]:
                    _max = (agent_action, _sum)

            # Save new payout and best response
            agent.best_response.set_value(action_dict, _max[0])
            new_function.set_value(action_dict, _max[1])

        # Delete all payout functions that involve the parent agent from all the dependant agents
        # And add the new payout functions to dependants
        for agent_name in dependant_agent_names:
            # Remove all functions that have agent_name in the dependants
            agents[agent_name].payout_functions = [function for function in agents[agent_name].payout_functions if
                                                   agent.name not in function.agent_names]
            if agent.name in agents[agent_name].dependant_agents:
                agents[agent_name].dependant_agents.remove(agent.name)

            agents[agent_name].payout_functions.append(new_function)

            # Add all dependants (except himself) to the agent's list if they are not already in
            agents[agent_name].dependant_agents.extend([agent for agent in dependant_agent_names if
                                                        agent != agent_name and agent not in agents[
                                                            agent_name].dependant_agents])

    # Second Pass, Reverse Order, excluding the last agent
    last_agent = list(elimination_agents)[-1]
    actions = {last_agent.name: str(last_agent.payout_functions[0].table.argmax().data[()])}
    for agent in list(elimination_agents)[-2::-1]:
        actions[agent.name] = agent.best_response.get_value(actions)

    print("\nVariable Elimination Result:")
    for key, value in sorted(actions.items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value), end=', ')
    return actions



agents = init_agents()
bf = brute_force(agents)
ve = variable_elimination(agents, order=["a1", "a2", "a3", "a4"])
