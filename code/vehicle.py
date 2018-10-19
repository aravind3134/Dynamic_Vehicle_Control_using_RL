"""Game class to represent Vehicle Simulator state."""

import numpy as np
from random import uniform
import time
import matplotlib.pyplot as plt

# Defined actions:
# 1. FORWARD: It takes two values - 0 for braking and
#                                   1 for moving forward with 0.1 m/s.
# 2. DIRECTION: It takes two values - 0 for straight
#                                     1 for right
# Defined States:
# [x, y, direction] represents the states that the RL Algorithm should traverse.
# The vehicle co-ordinates are assumed to be (0, 0) as of now.
# The direction is assumed to be 0.
# The initial state is fed from the system. The actions based on the current state determines the next state.
#     NEXT STATE = CURRENT STATE + RESULT OF FORWARD ACTION + RESULT OF DIRECTION ACTION
#
ACTION_NAMES = ["Forward", "RightLeft"]
ACTION_FORWARD = 0
ACTION_DIRECTION = 1
velocity = 0.1

class Vehicle_Controller(object):
  """ Initializes the Vehicle controller object"""
  def __init__(self, state, target_state , stop_simulation):
    self._target_state = target_state
    self.target_state_direction = target_state[2]
    target_state[2] = 0
    self.stop_simulation = stop_simulation
    if state is None and target_state is None:
      self._state = [0, 0, 0]
      self.add_random_point_to_follow()
    else:
      self._state = state
      self._target_state = target_state

  def copy(self):
    """Return a copy of self."""
    return Vehicle_Controller(np.copy(self._state), self._target_state, self.stop_simulation)

  def simulation_over(self):
    """Whether the simulation is over."""
    if self.stop_simulation == 1:
      return True
    else:
      return False

  def do_action(self, action, move_in_x):
    """ performs action on the current state. """
    temp_state = self._state
    reward = 0
    target_reached = self.goal_reached(self._state, self._target_state)
    if target_reached:
      reward = 1
    self.time_at_last_updation = time.time()
    if action == ACTION_FORWARD:
      if move_in_x == 1:
        forward_movement_x = self._result_of_action_forward_x()
        self._state[0] = temp_state[0] + forward_movement_x
      else:
        forward_movement_y = self._result_of_action_forward_y ()
        self._state[1] = temp_state[1] + forward_movement_y
    else:
      self._state[2] = temp_state[2] + self._result_of_action_direction()
    return reward

  def goal_reached(self, state, target_state):
    goal_reached = 0
    if int(state[0] - 10) < int(target_state[0]) <= int(state[0]):
      if int(state[1] - 10) < int(target_state[1]) <= int(state[1]):
        goal_reached = 1
    return goal_reached

  def _result_of_action_forward_x(self):
    forward_movement = velocity * time.clock()
    return forward_movement

  def _result_of_action_forward_y(self):
    forward_movement = velocity * time.clock()
    return forward_movement

  def _result_of_action_direction(self):
    direction = self._state[2] + 90 # degrees
    return direction

  def add_random_point_to_follow(self):
    """Add a random point to follow in the 2-D co-ordinate system."""
    x_pos = uniform(0.0, 100.0)
    y_pos = uniform (0.0, 100.0)
    direction = uniform(-180.0, 180.0)
    if x_pos is not self._state[0] and y_pos is not self._state[1] and direction is not self._state[2]:
      self._target_state = [x_pos, y_pos, direction]

  def state(self):
    """Return current state."""
    return self._state

  def print_state(self):
    """Prints the current state."""
    print(self._state)

  def print_target(self):
    print(self._target_state)

  def print_current_state_and_target(self):
    print("Current State: ", self._state, "\t", "Target: ", self._target_state)

def main():
    # see if the simulation points are ready to follow. If the flag is set, then the main function invokes.
    # if the data structure has the flag set to 1, then the object is invoked.
    # if datastructure.Monitor_Vehicle_using_DRL == 1:
    # state = [datastructure.x datastructure.y datastructure.direction]
    # target_state = [datastructure.target_x datastructure.target_y datastructure.target_direction]
    # else:
    # stop_simulation = 1
    stop_simulation = 0
    state = [0, 0, 0]
    target = [[7.04127043363047, 6.07706071711253, 127.05753848640688], [77.04127043363047, 67.07706071711253, 127.05753848640688], [177.04127043363047, 267.07706071711253, 127.05753848640688]]
    move_in_x = 0
    fig = plt.figure()
    plt.ion()
    ax = fig.add_subplot (111)
    for i in target:
      reward = 0
      target_state = i
      plt.plot (target_state[0], target_state[1], 'ro')
      a = Vehicle_Controller(state, target_state, stop_simulation)
      #a.print_current_state_and_target()
      count = 0
      while int(target_state[0]) >= int(a.state()[0]):
        #print(count)
        count = count + 1
        move_in_x = 1
        reward = a.do_action(ACTION_FORWARD, move_in_x)
        #a.print_current_state_and_target()
        ax.scatter (a.state ()[0], a.state ()[1])
        plt.pause (0.1)
      count = 0
      while int(target_state[1]) >= int(a.state()[1]):
        #print(count)
        count = count + 1
        move_in_x = 0
        reward = a.do_action(ACTION_FORWARD, move_in_x)
        #a.print_current_state_and_target()
        ax.scatter(a.state()[0], a.state()[1])
        plt.pause(0.1)
      # a.print_current_state_and_target()
      state = a.state()
      print("Reward: ", reward)
    plt.show()
if __name__ == "__main__":
    main()