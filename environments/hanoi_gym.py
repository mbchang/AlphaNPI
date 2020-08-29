from collections import namedtuple
import numpy as np
import pprint

from environments.hanoi_env import HanoiEnv

HanoiEnvState = namedtuple('HanoiEnvState', 
    ('pillars', 'roles', 'n', 'init_roles'))
HanoiGymDisk = namedtuple('HanoiGymDisk',
    ('pillars', 'roles', 'init_roles'))

"""
Notes
    need to make sure that self.max_steps < buffer size
        if buffer_size, then n <= 8
"""


class HanoiGym(HanoiEnv):
    def __init__(self, n):
        HanoiEnv.__init__(self, n=n)
        self.actions = {
            0: 'SWAP_S_A',
            1: 'SWAP_A_T',
            2: 'MOVE_DISK',
            3: 'STOP',
        }
        self.state_dim = 6
        self.action_dim = len(self.actions)
        self.max_steps = 10*2**n-1  # if n=8 then max_steps = 2560-1. If n=9 then max_steps = 5120-1

    def seed(self, seed):
        pass

    def render(self, mode):
        state = self.get_state()
        # create an array

        pass

    def reset(self):
        # print('self.tasks_dict.keys()', self.tasks_dict.keys())
        if len(self.tasks_dict) > 0:
            self.end_task()  # every time we reset, we force it to reset everything.
        try:
            observation, reparameterized_state = self.start_task(0)
        except AssertionError as error:
            assert error.args[0] in ['precondition not verified']
            self.end_task()
            observation, reparameterized_state = self.start_task(0)
        return reparameterized_state

    def step(self, action):
        """
            need to treat stop
        """
        old_observation, old_reparameterized_state = self.get_observation()
        old_reward = self.get_reward()
        old_done =  old_observation[-1]

        program = self.actions[action]
        if program == 'STOP':
            next_observation, next_reparameterized_state = self.get_observation()
            reward = self.get_reward()
            done = False#True
        else:
            try:
                next_observation, next_reparameterized_state = self.act(program)
            except AssertionError as error:
                assert error.args[0] in ['precondition not verified']
                next_observation, next_reparameterized_state = old_observation, old_reparameterized_state
                reward = old_reward
                done = old_done
            else:
                reward = self.get_reward()
                done = next_observation[-1]
        info = dict()
        return next_reparameterized_state, reward, done, info

    def get_state(self):
        state = HanoiEnv.get_state(self)
        state = HanoiEnvState(
            pillars=state[0],
            roles=state[1],
            n=state[2],
            init_roles=state[3])
        return state

    def reparameterize_state(self, state):
        assert isinstance(state, HanoiEnvState)
        assert state.init_roles == ['source', 'auxiliary', 'target']

        reparameterized_state = []  # must be ordered!
        for disk in range(state.n):
            disk_position_relative = [disk in pillar for pillar in state.pillars]
            disk_position_canonical = [disk_position_relative[state.roles.index(role)] for role in state.init_roles]
            roles = [state.init_roles.index(role) for role in state.roles]
            disk_representation = np.concatenate((disk_position_canonical, roles))
            reparameterized_state.append(disk_representation)
            # print('disk {} relative'.format(disk), disk_position_relative)
            # print('disk {} canonical'.format(disk), disk_position_canonical)
            # print('disk {} roles'.format(disk), roles)
            # print('disk {} disk_representation'.format(disk), disk_representation)
        return reparameterized_state

    def get_observation(self):
        observation = HanoiEnv.get_observation(self)
        state = self.get_state()
        reparameterized_state = np.stack(self.reparameterize_state(state), axis=0)
        return observation, reparameterized_state

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
    print('Obs: {} {} Action: {} Next Obs: {} Reward: {} Done: {} Info: {}'.format(
        obs, actions[action], action, next_obs, reward, done, info))


def execute_transition(obs, action, env, obs_action_pairs):
    """
        (1, 0, 0, 0, 0) can be avoided if we had an additional action SWAP_S_T that can be used between task boundaries
        (1, 0, 1, 0, 0) can be avoided if we include the roles of the pillars in the state, plus the n %2 == 0?
            [1 0 1 0 0], ['target', 'auxiliary', 'source'] --> 1  # actually unnecessary
            [1 0 1 0 0], ['source', 'target', 'auxiliary'] --> 1 *
            [1 0 1 0 0], ['auxiliary', 'target', 'source'] --> 1
            [1 0 1 0 0], ['target', 'source', 'auxiliary'] --> 1

            [1 0 1 0 0], ['source', 'auxiliary', 'target'] --> 2
            [1 0 1 0 0], ['target', 'auxiliary', 'source'] --> 2
            [1 0 1 0 0]  ['source', 'target', 'auxiliary'] --> 2 *

        Hmm, I wonder how to do this without encoding the state. I could just encode the state though I suppose.

    """
    # if tuple(obs) == (1, 0, 0, 0, 0, 0, 2, 1, 1):
    #     state = env.get_state()
    #     print('Observation: {} State: {} Action: {}'.format(obs, state, action))
    # if tuple(obs) == (1, 0, 1, 0, 0, 0, 2, 1, 1):
    #     state = env.get_state()
    #     print('Observation: {} State: {} Action: {}'.format(obs, state, action))
    # if tuple(obs) == (1, 1, 1, 0, 0):
    #     state = env.get_state()
    #     print('Observation: {} State: {} Action: {}'.format(obs, state, action))   


    # print('Observation: {} State: {} Action: {}'.format(obs, state, action))   

    next_obs, reward, done, info = env.step(action)
    visualize_transition(obs, action, next_obs, reward, done, info)
    print('State: {}'.format(env.get_state()))
    hashable_obs = tuple([tuple(disk) for disk in obs])
    if hashable_obs not in obs_action_pairs:
        obs_action_pairs[hashable_obs] = set()
    obs_action_pairs[hashable_obs].add(env.actions[action])
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
        print('State: {}'.format(env.get_state()))
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2, help='number of disks')
    args = parser.parse_args()
    test_hanoi_gym_n(max_n=args.n)
