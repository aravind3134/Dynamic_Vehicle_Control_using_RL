"""Algorithms and strategies to play 2048 and collect experience."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from vehicle import Vehicle_Controller
# pylint: disable=too-many-arguments,too-few-public-methods
class Experience(object):
  """Struct to encapsulate the experience of a single turn."""

  def __init__(self, state, action, reward, next_state, game_over):
    """Initialize Experience

    Args:
      state: Shape (4, 4) numpy array, the state before the action was executed
      action: Number in range(4), action that was taken
      reward: Number, experienced reward
      next_state: Shape (4, 4) numpy array, the state after the action was
          executed
      game_over: boolean, whether the
    """
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.game_over = game_over



def Follow(strategy, verbose=False):
  """Plays a single game, using a provided strategy.

  Args:
    strategy: A function that takes as argument a state and a list of available
        actions and returns an action from the list.
    allow_unavailable_action: Boolean, whether strategy is passed all actions
        or just the available ones.
    verbose: If true, prints game states, actions and scores.

  Returns:
    score, experiences where score is the final score and experiences is the
        list Experience instances that represent the collected experience.
  """
  # eventually to be removed
  '''
  csv_path = 'C:/Users/Aravind/Documents/NCSU/3rd sem/ECE 542/project06/'
  os.chdir (csv_path)
  file_name = None
  for i in glob.glob('*.*'):
      if i == 'odom_simplified.csv':
          file_name  = i
          break
  with open (file_name) as f:
      a = csv.reader (f)
  '''
  initial_state = [6.436814971,	6.613150112	, -0.526185188,	0.154138194]
  final_state = [6.437283741,	6.613119713,	-0.526327424,	0.153992516]
  controller = Vehicle_Controller(initial_state, final_state, stop_simulation = False)
  print("1st checkpoint")
  state = controller.state()
  print ("2nd checkpoint")
  print("Current state: ", state)
  game_over = controller.simulation_over()
  print ("3rd checkpoint")
  print("Is simulation over?", True if game_over == 1 else False)
  experiences = []
  print("Just before entering for loop")
  while not game_over:
    if verbose:
      controller.print_state()

    old_state = state
    #print("Tell me you are here in play.py")
    next_action = strategy(
        old_state, range(15))
    print(next_action)
    #print ("Tell me you are here in stage 2 in play.py")
    reward = controller.do_action(next_action)
    print("Reward", reward)
    state = controller.state()
    game_over = controller.simulation_over()

    if verbose:
      print("Action:", ACTION_NAMES[next_action])
      print("Reward:", reward)
    experiences.append(Experience(old_state, next_action, reward, state, False))
  return experiences


def random_strategy(_, actions):
  """Strategy that always chooses actions at random."""
  return np.random.choice(actions)

def static_preference_strategy(_, actions):
  """Always prefer left over up over right over top."""
  return actions

def highest_reward_strategy(state, actions):
  """Strategy that always chooses the action of highest immediate reward.

  If there are any ties, the strategy prefers left over up over right over down.
  """
  sorted_actions = np.sort(actions)[::-1]
  rewards = [Vehicle_Controller(np.copy(state)).do_action(action)
             for action in sorted_actions]
  action_index = np.argsort(rewards)[-1]
  return sorted_actions[action_index]

def make_greedy_strategy(get_q_values, verbose=False):
  """Makes greedy_strategy."""

  def greedy_strategy(state, actions):
    """Strategy that always picks the action of maximum Q(state, action)."""
    q_values = get_q_values(state)
    if verbose:
      print("State:")
      print(state)
      print("Q-Values:")
      for action, q_value, action_name in zip(range(2), q_values, ACTION_NAMES):
        not_available_string = "" if action in actions else "(not available)"
        print("%s:\t%.2f %s" % (action_name, q_value, not_available_string))
    sorted_actions = np.argsort(q_values)
    action = [a for a in sorted_actions if a in actions][-1]
    if verbose:
      print("-->", ACTION_NAMES[action])
    return action

  return greedy_strategy


def make_epsilon_greedy_strategy(get_q_values, epsilon):
  """Makes epsilon_greedy_strategy."""

  greedy_strategy = make_greedy_strategy(get_q_values)

  def epsilon_greedy_strategy(state, actions):
    """Picks random action with prob. epsilon, otherwise greedy_strategy."""
    do_random_action = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
    if do_random_action:
      return random_strategy(state, actions)
    return greedy_strategy(state, actions)

  return epsilon_greedy_strategy
