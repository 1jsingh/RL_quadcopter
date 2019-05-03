import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 5

        self.reset()

        self.state_size = self.state.size
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.t = 0
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self,done,rotor_speeds,in_bounds):
        """Uses current pose of sim to return reward."""
        
        penalty = 10

        # takeoff task
        reward = 1 - penalty*(not in_bounds) -(1e-2)*(np.linalg.norm(self.sim.pose[:3] - self.target_pos)) #- (1e-4)*(np.sum(self.sim.pose[3:]**2)+np.sum(self.sim.angular_v**2))
        
        # float task
        # reward = 1 - penalty*(not in_bounds) - (1e-4)*(np.sum(self.sim.pose[3:]**2)+np.sum(self.sim.angular_v**2))

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        #pose_all = []
        for _ in range(self.action_repeat):
            self.t += 1
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward(done,rotor_speeds,self.sim.in_bounds) 
            #pose_all.append(self.sim.pose)
        #next_state = np.concatenate(pose_all)
        next_state = np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v))
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.t = 0
        self.sim.reset()
        #self.state = np.concatenate([self.sim.pose] * self.action_repeat)
        self.state = np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v))
        return self.state