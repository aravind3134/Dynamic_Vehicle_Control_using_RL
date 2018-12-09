"""Vehicle_Controller class to represent Vehicle Simulator state."""

import numpy as np
from random import uniform
import time, math
import matplotlib.pyplot as plt

# The vehicle moves every time with a constant velocity.
# Defined actions:
# 1. BASED ON DIRECTION: bins of 20 between [100, 400]
# Actions: 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 370, 390
# The values of raw steering angle sensor value from ADC is passed and the expected values of refined raw steering angle
# sensor values are returned to the algorithm from Deep Reinforcement Learning algorithm.
# Defined States:
# [x, y, direction] represents the states that the RL Algorithm should traverse.
# The vehicle co-ordinates are assumed to be (0, 0) as of now.
# The direction is assumed to be 0.
# The initial state is fed from the system. The actions based on the current state determines the next state.
#     NEXT STATE = CURRENT STATE + RESULT OF DIRECTION ACTION
#
ACTION_NAMES = ["100", "120", "140", "160", "180", "200", "220", "240", "260", "280", "300", "320", "340", "360", "380", "400"]
ACTION_100 = 0
ACTION_120 = 1
ACTION_140 = 2
ACTION_160 = 3
ACTION_180 = 4
ACTION_200 = 5
ACTION_220 = 6
ACTION_240 = 7
ACTION_260 = 8
ACTION_280 = 9
ACTION_300 = 10
ACTION_320 = 11
ACTION_340 = 12
ACTION_360 = 13
ACTION_380 = 14
ACTION_400 = 15

NUM_OF_ACTIONS = len(ACTION_NAMES)
velocity = 1
ACT_STEERING_MAX_ANGLE_RANGE = math.pi/2
SMALL = 1e-8
INCHES2METERS = 25.4/1000.0
VEHICLEWHEEL2WHEELLENGTH = 72*INCHES2METERS

class Vehicle_Controller(object):
  """ Initializes the Vehicle controller object"""
  def __init__(self, state, target_state , stop_simulation):
    print ("Vehicle Controller Object is created with state: ", state, "and target state: ", target_state)
    self._target_state = target_state
    self.stop_simulation = stop_simulation
    if state is None and target_state is None:
        self._state = [0, 0, 0, 0]
        self.add_random_point_to_follow()
    else:
        self._state = state
        self._target_state = target_state
        self.LastUpdateTime = time.time()

  def copy(self):
    """Return a copy of self."""
    return Vehicle_Controller(np.copy(self._state), self._target_state, self.stop_simulation)

  def simulation_over(self):
    """Whether the simulation is over."""
    if self.stop_simulation == 1:
      return True
    else:
      return False

  def do_action(self, action):
    """ performs action on the current state. """
    reward = 0
    temp_state = self._state
    print("Current state" ,temp_state)
    print(action)
    steer_action = (math.pi/4) + ((int(ACTION_NAMES[action])-100)/(400-100))*(-math.pi/2)
    timeNow = time.time()
    deltaT = timeNow - self.LastUpdateTime

    steering = temp_state[2] + action

    if (steering > ACT_STEERING_MAX_ANGLE_RANGE / 2):
        temp_state[2]= ACT_STEERING_MAX_ANGLE_RANGE / 2
    if (steering < -ACT_STEERING_MAX_ANGLE_RANGE / 2):
        temp_state[2] = -ACT_STEERING_MAX_ANGLE_RANGE / 2

    tan_steer = math.tan(steering)
    if (abs(tan_steer) < SMALL):
    	tan_steer = SMALL

    sin_steer = math.sin(steering);
    if (abs(sin_steer) < SMALL):
    	sin_steer = SMALL;

    backradius = VEHICLEWHEEL2WHEELLENGTH/tan_steer
    deltalength = velocity*deltaT
    delta_arc_angle = deltalength / backradius
    centerx = temp_state[0] - math.sin(temp_state[3])*backradius
    centery = temp_state[1] + math.cos(temp_state[3])*backradius

    new_heading = temp_state[3] + delta_arc_angle
    temp_state[0] = centerx + math.sin(new_heading)*backradius
    temp_state[1] = centery - math.cos(new_heading)*backradius

    if (new_heading > math.pi):
		temp_state[3] -= 2*math.pi

    if (new_heading< -math.pi):
        temp_state[3] += 2*math.pi


    self.LastUpdateTime = timeNow

    target_reached = self.goal_reached(temp_state, self._target_state)
    if target_reached:
        reward = 1
    print("Next state" ,temp_state)
    self._state = temp_state
    return reward

  def goal_reached(self, state, target_state):
    goal_reached = 0
    if (state[0] - 1) < (target_state[0]) <= (state[0]+1):
      if (state[1] - 1) < (target_state[1]) <= (state[1]+1):
          if (state[2] - 0.1) < (target_state[2]) <= (state[2]+ 0.1):
            if (state[3] - 0.1) < (target_state[3]) <= (state[3]+ 0.1):
                  goal_reached = 1
    return goal_reached

  def add_random_point_to_follow(self):
    """Add a random point to follow in the 2-D co-ordinate system."""
    x_pos = uniform(-50.0, 50.0)
    y_pos = uniform (-50.0, 50.0)
    direction = uniform(-45.0, 45.0)
    steer_direction = uniform(-45.0, 45.0)
    if x_pos is not self._state[0] and y_pos is not self._state[1] and direction is not self._state[2] and steer_direction is not self._state[3]:
      self._target_state = [x_pos, y_pos, direction, steer_direction]

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


#  --------------------------------------------------- TO BE REMOVED -------------------------------------------------#
'''
  def _result_of_action_forward_x(self):
    forward_movement = velocity * time.clock()
    return forward_movement

  def _result_of_action_forward_y(self):
    forward_movement = velocity * time.clock()
    return forward_movement

  def _result_of_action_direction(self):
    direction = self._state[2] # degrees
    return direction
'''

'''
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
      plt.plot (target_state[0], target_state[1], 'rx')
      a = Vehicle_Controller(state, target_state, stop_simulation)
      #a.print_current_state_and_target()
      count = 0
      while int(target_state[0]) >= int(a.state()[0]):
        #print(count)
        count = count + 1
        reward = a.do_action(ACTION_110)
        #a.print_current_state_and_target()
        ax.scatter (a.state ()[0], a.state ()[1])
        plt.pause (0.1)
      count = 0
      while int(target_state[1]) >= int(a.state()[1]):
        #print(count)
        count = count + 1
        move_in_x = 0
        reward = a.do_action(ACTION_110, move_in_x)
        #a.print_current_state_and_target()
        ax.scatter(a.state()[0], a.state()[1])
        plt.pause(0.1)
      # a.print_current_state_and_target()
      state = a.state()
      print("Reward: ", reward)
    plt.show()
'''
