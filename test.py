import copy
import random
import time

from VariableElimination import *

NUM_TESTS = 10

NUM_AGENTS = 13
LINK_CHANCE = 0.15

# Used if NUM_AGENTS is None
MAX_AGENTS = 25
MIN_AGENTS = 4

def random_data(link_chance=LINK_CHANCE, num_agents=None):
    data = {"agents": [], "connections": []}
    # Create random number of agents
    if num_agents is None:
        num_agents = random.randint(MIN_AGENTS, MAX_AGENTS)
    for num in range(num_agents):
        data["agents"].append("a" + str(num))

    # Create random non duplicate connections based on link_chance
    for first_agent in data["agents"]:
        for second_agent in data["agents"]:
            if first_agent != second_agent and random.randint(0, 100) < link_chance * 100 and [second_agent,
                                                                                               first_agent] not in data[
                "connections"]:
                data["connections"].append([first_agent, second_agent])

    # If there is an agent that in not connected, randomly connect that agent.
    for agent in data["agents"]:
        if not any([agent == pair[0] or agent == pair[1] for pair in data["connections"]]):
            choice = [agent, random.choice([a for a in data["agents"] if a != agent])]
            while choice in data["connections"] or [choice[1], choice[0]] in data["connections"]:
                choice = [agent, random.choice([a for a in data["agents"] if a != agent])]
            data["connections"].append(choice)
    return data


def random_agents(link_chance=LINK_CHANCE, num_agents=None):
    data = random_data(link_chance=link_chance, num_agents=num_agents)

    agents = {}
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


def brute_force(agents, debug=False, locked_actions={}):
    max = ("", -1)
    agent_names = [agents[x].name for x in agents]

    res = []
    for agent in agents.values():
        if agent.name in locked_actions.keys():
            res.append([locked_actions[agent.name]])
        else:
            res.append(agent.possible_actions)
    action_product = list(product(*res))

    all_payouts = list(
        dict.fromkeys([item for sublist in [agent.payout_functions for agent in agents.values()] for item in sublist]))
    for joint_action in action_product:
        action_dict = {agent_names[i]: joint_action[i] for i in range(len(agent_names))}
        _sum = 0
        for payout in all_payouts:
            _sum += payout.get_value(action_dict)
        if _sum > max[1]:
            max = action_dict, _sum
    if debug:
        print("\nBest joint action:")
        for key, value in sorted(max[0].items(), key=lambda x: x[0]):
            print("{} : {}".format(key, value), end=', ')
    return max[0]


def test_base(agents, debug=False, locked_actions={}):
    bf_time = 0
    ve_time = 0

    start = time.time()
    ve = variable_elimination(copy.deepcopy(agents), locked_actions=copy.deepcopy(locked_actions), debug=debug)
    ve_time += time.time() - start
    start = time.time()
    bf = brute_force(copy.deepcopy(agents), locked_actions=copy.deepcopy(locked_actions), debug=debug)
    bf_time += time.time() - start
    if bf != ve:
        # ve = variable_elimination(copy.deepcopy(agents), locked_actions=copy.deepcopy(locked_actions), debug=debug)
        # bf = brute_force(copy.deepcopy(agents), locked_actions=copy.deepcopy(locked_actions), debug=debug)
        return False, bf_time, ve_time

    return True, bf_time, ve_time


def test(debug=False, lock_chance=0.5, num_agents=NUM_AGENTS, locked_actions=False):
    num_failed = 0
    bf_time = 0
    ve_time = 0
    data_time= 0
    for i in range(NUM_TESTS):
        start = time.time()
        agents = random_agents(num_agents=num_agents)
        data_time = time.time() - start
        if locked_actions:
            actions = {agent: random.choice(agents[agent].possible_actions) for agent in agents if
                              random.randint(0, 100) < lock_chance * 100}
            if len(actions) == len(agents):
                actions.pop(random.choice(list(actions.keys())))
        else:
            actions = {}
        passed, _bf_time, _ve_time = test_base(agents, locked_actions=actions, debug=debug)
        bf_time += _bf_time
        ve_time += _ve_time
        if not passed:
            num_failed += 1

    print("Tests Completed:", NUM_TESTS, "Tests Failed:", num_failed)
    print("Data Creation Time:", data_time, "\nBrute Force time:", bf_time, "\nVariable Elimination Time:", ve_time)


print("Normal test (no locked actions):")
test()

print("\nTest with locked actions:")
test(locked_actions=True)

