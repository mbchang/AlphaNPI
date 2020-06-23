import pprint

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
        self.state_dim = 5
        self.action_dim = len(self.actions)
        self.max_steps = 5*2**n-1

    def seed(self, seed):
        pass

    def render(self, mode):
        pass

    def reset(self):
        try:
            observation = self.start_task(0)
        except AssertionError as error:
            assert error.args[0] in ['precondition not verified']
            self.end_task()
            observation = self.start_task(0)
        return observation

    def step(self, action):
        """
            need to treat stop
        """
        old_observation = self.get_observation()
        old_reward = self.get_reward()
        old_done =  old_observation[-1]

        program = self.actions[action]
        if program == 'STOP':
            next_observation = self.get_observation()
            reward = self.get_reward()
            done = True
        else:
            try:
                next_observation = self.act(program)
            except AssertionError as error:
                assert error.args[0] in ['precondition not verified']
                next_observation = old_observation
                reward = old_reward
                done = old_done
            else:
                reward = self.get_reward()
                done = next_observation[-1]
        info = dict()
        return next_observation, reward, done, info

"""
Testing
"""
def visualize_reset(obs):
    print('Initial Observation: {}'.format(obs))

def visualize_transition(obs, action, next_obs, reward, done, info):
    actions = {
            0: 'SWAP_S_A',
            1: 'SWAP_A_T',
            2: 'MOVE_DISK',
            3: 'STOP',
        }
    # print('Obs: {} {} Action: {} Next Obs: {} Reward: {} Done: {} Info: {}'.format(
    #     obs, actions[action], action, next_obs, reward, done, info))


def execute_transition(obs, action, env, obs_action_pairs):
    next_obs, reward, done, info = env.step(action)
    visualize_transition(obs, action, next_obs, reward, done, info)
    if tuple(obs) not in obs_action_pairs:
        obs_action_pairs[tuple(obs)] = set()
    obs_action_pairs[tuple(obs)].add(env.actions[action])
    obs = next_obs
    return obs, obs_action_pairs

def execute_n_policy(obs, n, env, obs_action_pairs):
    if n == 1:
        policy = [2]
    else:
        n_minus_1_policy = lambda obs, env, obs_action_pairs: execute_n_policy(obs, n-1, env, obs_action_pairs)
        policy = [1, n_minus_1_policy, 1, 2, 0, n_minus_1_policy, 0]
    for action in policy:
        if isinstance(action, int):
            obs, obs_action_pairs = execute_transition(obs, action, env, obs_action_pairs)
        else:
            obs, obs_action_pairs = action(obs, env, obs_action_pairs)
    return obs, obs_action_pairs

def test_hanoi_gym_n(max_n):
    for n in range(1, max_n+1):
        print('n = {}'.format(n))
        obs_action_pairs = dict()

        env = HanoiGym(n=n)
        obs = env.reset()
        visualize_reset(obs)
        # obs, obs_action_pairs = execute_n4_policy(obs, env, obs_action_pairs)
        obs, obs_action_pairs = execute_n_policy(obs, n, env, obs_action_pairs)
        pprint.pprint(obs_action_pairs)


"""
n = 1
Initial Observation: [1 0 1 1 0]
{(1, 0, 1, 1, 0): {'MOVE_DISK'}}
n = 2
Initial Observation: [1 0 1 0 0]
{(0, 0, 0, 0, 1): {'SWAP_S_A'},
 (0, 1, 0, 0, 0): {'SWAP_S_A'},
 (0, 1, 1, 0, 0): {'MOVE_DISK'},
 (1, 0, 0, 0, 0): {'SWAP_A_T'},
 (1, 0, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'}}
n = 3
Initial Observation: [1 0 1 0 0]
{(0, 0, 0, 0, 0): {'SWAP_S_A'},
 (0, 0, 0, 0, 1): {'SWAP_S_A'},
 (0, 1, 0, 0, 0): {'SWAP_S_A'},
 (0, 1, 1, 0, 0): {'MOVE_DISK'},
 (1, 0, 0, 0, 0): {'SWAP_A_T'},
 (1, 0, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'},
 (1, 1, 1, 0, 0): {'MOVE_DISK'}}
n = 4
Initial Observation: [1 0 1 0 0]
{(0, 0, 0, 0, 0): {'SWAP_S_A'},
 (0, 0, 0, 0, 1): {'SWAP_S_A'},
 (0, 1, 0, 0, 0): {'SWAP_S_A'},
 (0, 1, 1, 0, 0): {'MOVE_DISK'},
 (1, 0, 0, 0, 0): {'SWAP_S_A', 'SWAP_A_T'},
 (1, 0, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'},
 (1, 1, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'}}
n = 5
Initial Observation: [1 0 1 0 0]
{(0, 0, 0, 0, 0): {'SWAP_S_A'},
 (0, 0, 0, 0, 1): {'SWAP_S_A'},
 (0, 1, 0, 0, 0): {'SWAP_S_A'},
 (0, 1, 1, 0, 0): {'MOVE_DISK'},
 (1, 0, 0, 0, 0): {'SWAP_S_A', 'SWAP_A_T'},
 (1, 0, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'},
 (1, 1, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'}}
n = 6
Initial Observation: [1 0 1 0 0]
{(0, 0, 0, 0, 0): {'SWAP_S_A'},
 (0, 0, 0, 0, 1): {'SWAP_S_A'},
 (0, 1, 0, 0, 0): {'SWAP_S_A'},
 (0, 1, 1, 0, 0): {'MOVE_DISK'},
 (1, 0, 0, 0, 0): {'SWAP_S_A', 'SWAP_A_T'},
 (1, 0, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'},
 (1, 1, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'}}
n = 7
Initial Observation: [1 0 1 0 0]
{(0, 0, 0, 0, 0): {'SWAP_S_A'},
 (0, 0, 0, 0, 1): {'SWAP_S_A'},
 (0, 1, 0, 0, 0): {'SWAP_S_A'},
 (0, 1, 1, 0, 0): {'MOVE_DISK'},
 (1, 0, 0, 0, 0): {'SWAP_S_A', 'SWAP_A_T'},
 (1, 0, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'},
 (1, 1, 1, 0, 0): {'MOVE_DISK', 'SWAP_A_T'}}
"""


if __name__ == '__main__':
    test_hanoi_gym_n(max_n=7)
