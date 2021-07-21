import random
import numpy as np
from random import choice

class Agent:

    def __init__(self, name):
        self.name = name
        self.possible_actions = ["0", "1"]
        self.possible_states = ["A", "B"]
        self.current_state = choice(self.possible_states)
        self.payout_functions = []
        self.dependant_agents = []
        self.q_tables = {}
