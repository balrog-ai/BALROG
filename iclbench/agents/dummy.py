import logging
from collections import defaultdict, namedtuple

from iclbench.agents.base import BaseAgent

def make_dummy_action(text):
    return namedtuple('Action', 'choices')([
        namedtuple('Choice', 'message')(
            namedtuple('Message', 'content')(text)
        )
    ])

class DummyAgent(BaseAgent):
    """
    For debugging.
    """
    
    def __init__(self):
        super().__init__()

    def act(self, obs, prev_action=None):
        if prev_action:
            self.action_history.append(prev_action)
            self.action_frequency[prev_action] += 1
        return make_dummy_action("dummy_action")
