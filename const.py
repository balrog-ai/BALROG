from nle_language_wrapper import NLELanguageWrapper
from nle.nethack import ACTIONS

ACTION_NAMES = [action_strs[0] for action, action_strs in NLELanguageWrapper.all_nle_action_map.items() if action in ACTIONS]
ACTION_OPTIONS = '\n'.join([f"{i}) {action_name}" for i, action_name in enumerate(ACTION_NAMES)])

print(ACTION_OPTIONS)