from environments.hanoi_env import HanoiEnv

class HanoiGym(HanoiEnv):
    def __init__(self, n):
        HanoiEnv.__init__(self, n=n)
        self.actions = {
            0: 'SWAP_S_A',
            1: 'SWAP_A_T',
            2: 'MOVE_DISK',
            3: 'STOP',
        }

    def reset(self):
        observation = self.start_task(0)
        return observation

    def step(self, action):
        """
            need to treat stop
        """
        program = self.actions[action]
        if program == 'STOP':
            observation = self.get_observation()
            reward = self.get_reward()
            done = True
        else:
            observation = self.act(program)
            reward = self.get_reward()
            done = observation[-1]
        info = dict()
        return observation, reward, done, info

"""
Testing
"""
def visualize_reset(obs):
    print('Initial Observation: {}'.format(obs))

def visualize_transition(next_obs, reward, done, info):
    print('Next Observation: {}'.format(next_obs))
    print('Reward: {}'.format(reward))
    print('Done: {}'.format(done))
    print('Info: {}'.format(info))

def test_hanoi_gym_1():
    env = HanoiGym(n=1)
    obs = env.reset()
    visualize_reset(obs)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

def test_hanoi_gym_2():
    env = HanoiGym(n=2)
    obs = env.reset()
    visualize_reset(obs)

    action = 1  # SWAP_A_T
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 1  # SWAP_A_T
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 0  # SWAP_S_A
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 0  # SWAP_S_A
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

def test_hanoi_gym_3():
    env = HanoiGym(n=3)
    obs = env.reset()
    visualize_reset(obs)

    action = 1  # SWAP_A_T
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 1  # SWAP_A_T
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 1  # SWAP_A_T
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 0  # SWAP_S_A
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 0  # SWAP_S_A
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 1  # SWAP_A_T
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 0  # SWAP_S_A
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 1  # SWAP_A_T
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 1  # SWAP_A_T
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 0  # SWAP_S_A
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 2  # MOVE_DISK
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 0  # SWAP_S_A
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)

    action = 0  # SWAP_S_A
    next_obs, reward, done, info = env.step(action)
    visualize_transition(next_obs, reward, done, info)




if __name__ == '__main__':
    # test_hanoi_gym_1()
    # test_hanoi_gym_2()
    test_hanoi_gym_3()