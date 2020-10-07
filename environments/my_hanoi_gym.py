from collections import namedtuple
import copy
import numpy as np
import pprint

from environments.hanoi_env import HanoiEnv

HanoiEnvState = namedtuple('HanoiEnvState', 
    ('pillars', 'roles', 'n', 'init_roles'))


"""
Notes
    need to make sure that self.max_steps < buffer size
        if buffer_size, then n <= 8
"""


class MyHanoiGym():
    def __init__(self, n):
        self.pillar_map = {
            0: 'source',
            1: 'auxiliary',
            2: 'target',
        }
        self.pillar_rmap = {v: k for k,v in self.pillar_map.items()}

        self.n = n
        self.actions = {
            0: (0, 1),
            1: (0, 2),
            2: (1, 2),
            3: (1, 0),
            4: (2, 0),
            5: (2, 1),
        }
        self.pillars = ([], [], [])
        self.roles = []
        self.init_roles = []

    def seed(self, seed):
        pass  # will be used to randomize the source, auxiliary, target

    def render(self, mode):
        state = self.get_state()

        # create a mapping between rectangles and disk ids.




        # create an array

        pass

    def text_render(self):
        state = self.get_state()
        pillars = state.pillars
        n = state.n
        roles = [s[0] for s in state.roles]

        disk_id_to_width = lambda d: 1+2*d
        disk_id_to_start_idx = lambda d: (max_disk_diameter-disk_id_to_width(d))//2

        max_disk_diameter = disk_id_to_width(n-1)
        max_pillar_height = n
        text_height = 1
        padding = 1
        width = padding + 3*(max_disk_diameter + padding)  # left --> right
        height = padding + max_pillar_height + padding + text_height + padding # top --> bottom

        top_border = '-'*width
        middle_border = '-'*width
        bottom_border = '-'*width

        def replace_index_in_string(string, character, index):
            return string[:index] + character + string[index+1:]

        text_canvas = [' '*max_disk_diameter for _ in roles]
        for i, role in enumerate(roles):
            text_canvas[i] = replace_index_in_string(text_canvas[i], role, disk_id_to_start_idx(0))

        text = '|{}|{}|{}|'.format(*text_canvas)

        def generate_pillar_array(pillar):
            pillar_array = np.zeros((max_pillar_height, max_disk_diameter))
            for i, disk in enumerate(pillar):
                start_idx = disk_id_to_start_idx(disk)
                pillar_array[max_pillar_height-i-1, start_idx:start_idx+disk_id_to_width(disk)] = 1
            return pillar_array

        pillar_arrays = np.concatenate([generate_pillar_array(pillar) for pillar in pillars], axis=-1)

        def pillar_arrays_to_string(p_arrays):
            assert p_arrays.shape[1] % 3 == 0  # 3 pillars
            p_strings = []
            for row in p_arrays:
                p_array_row = ''.join(['*' if e == 1 else ' ' for e in row])
                p_array_split = [p_array_row[i:i+max_disk_diameter] for i in range(0, p_arrays.shape[1], max_disk_diameter)]
                p_strings.append('|{}|{}|{}|'.format(*p_array_split))
            return p_strings

        pillar_strings = pillar_arrays_to_string(pillar_arrays)

        s = '\n'.join([
            top_border,
            *pillar_strings,
            middle_border,
            text,
            bottom_border])
        return s

    def reset(self):
        self.roles = ['source', 'auxiliary', 'target']
        self.init_roles = self.roles.copy()
        # can decide if we want to shuffle the roles or not

        src_pos = self.roles.index('source')
        self.pillars = ([], [], [])
        for i in range(self.n-1, -1, -1):
            self.pillars[src_pos].append(i)

        state = self.get_state()
        reparameterized_state = np.stack(self._reparameterize_state(state), axis=0)
        return reparameterized_state

    def step(self, action):
        start_pillar_idx = self.actions[action][0]
        end_pillar_idx = self.actions[action][1]
        old_state = self.get_state()
        if self._is_move_possible(self.pillars[start_pillar_idx], self.pillars[end_pillar_idx]):
            disk = self._pop(start_pillar_idx)
            self._push(end_pillar_idx, disk)
            state = self.get_state()
            done = self.get_done()
            reward = self.get_reward()
        else:
            state = old_state
            done = self.get_done()
            reward = -0.01
        reparameterized_state = np.stack(self._reparameterize_state(state), axis=0)
        info = dict()
        return reparameterized_state, reward, done, info

    def get_state(self):
        state = HanoiEnvState(
            pillars=copy.deepcopy(self.pillars),
            roles=copy.deepcopy(self.roles),
            n=self.n,
            init_roles=copy.deepcopy(self.init_roles)
            )
        return state

    def get_reward(self):
        reward = self._is_solved()
        return reward

    def get_done(self):
        done = self._is_solved()
        return done

    def _reparameterize_state(self, state):
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

    def _pop(self, i):
        """Take a disk off the top of pillars[i] and return it"""
        if len(self.pillars[i]) > 0:
            return self.pillars[i].pop()
        else:
            raise EmptyTowerException("Tried to pull a disk off a pillar which is empty")

    def _push(self, i, disk):
        """Put a disk on top of pillars[i]"""
        if len(self.pillars[i]) == 0 or self.pillars[i][-1] > disk:
            self.pillars[i].append(disk)
        else:
            raise InvertedTowerException("Tried to put larger disk on smaller disk")

    def _is_move_possible(self, start_pillar, end_pillar):
        if len(start_pillar) == 0:
            return False
        elif len(end_pillar) == 0:
            return True
        else:
            is_possible = len(start_pillar) >= 1
            if is_possible:
                is_possible &= end_pillar[-1] > start_pillar[-1]
            return is_possible

    def _is_solved(self):
        targ_pos = self.init_roles.index('target')
        return self.pillars[targ_pos] == range(self.n)



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
        # state = self.get_state()
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

# def visualize_transition(obs, action, next_obs, reward, done, info):
#     actions = {
#             0: 'SWAP_S_A',
#             1: 'SWAP_A_T',
#             2: 'MOVE_DISK',
#             3: 'STOP',
#         }
#     print('Obs: {} {} Action: {} Next Obs: {} Reward: {} Done: {} Info: {}'.format(
#         obs, actions[action], action, next_obs, reward, done, info))


# def execute_transition(obs, action, env, obs_action_pairs):
#     next_obs, reward, done, info = env.step(action)
#     visualize_transition(obs, action, next_obs, reward, done, info)
#     print('State: {}'.format(env.get_state()))
#     hashable_obs = tuple([tuple(disk) for disk in obs])
#     if hashable_obs not in obs_action_pairs:
#         obs_action_pairs[hashable_obs] = set()
#     obs_action_pairs[hashable_obs].add(env.actions[action])
#     obs = next_obs
#     return obs, obs_action_pairs

# def execute_n_policy(obs, n, env, obs_action_pairs):
#     if n == 1:
#         policy = [1]
#     else:
#         n_minus_1_policy = lambda obs, env, obs_action_pairs: execute_n_policy(obs, n-1, env, obs_action_pairs)
#         policy = [1, n_minus_1_policy, 1, 2, 0, n_minus_1_policy, 0]
#     for action in policy:
#         if isinstance(action, int):
#             obs, obs_action_pairs = execute_transition(obs, action, env, obs_action_pairs)
#         else:
#             obs, obs_action_pairs = action(obs, env, obs_action_pairs)
#     return obs, obs_action_pairs

# def test_hanoi_gym_n(max_n):
#     for n in range(1, max_n+1):

#         n = 1

#         print('n = {}'.format(n))
#         obs_action_pairs = dict()

#         env = MyHanoiGym(n=n)
#         obs = env.reset()
#         print('State: {}\n{}'.format(env.get_state(), env.text_render()))
#         obs, obs_action_pairs = execute_n_policy(obs, n, env, obs_action_pairs)
#         pprint.pprint(obs_action_pairs)

def visualize_transition(obs, action, next_obs, reward, done, info):
    actions = {
            0: (0, 1),
            1: (0, 2),
            2: (1, 2),
            3: (1, 0),
            4: (2, 0),
            5: (2, 1),
        }
    print('Obs: {} Action: {} Next Obs: {} Reward: {} Done: {} Info: {}'.format(
        obs, actions[action], next_obs, reward, done, info))

def execute_transition(obs, action, env, obs_action_pairs):
    next_obs, reward, done, info = env.step(action)
    visualize_transition(obs, action, next_obs, reward, done, info)
    print('State: {}\n{}'.format(env.get_state(), env.text_render()))
    hashable_obs = tuple([tuple(disk) for disk in obs])
    if hashable_obs not in obs_action_pairs:
        obs_action_pairs[hashable_obs] = set()
    obs_action_pairs[hashable_obs].add(env.actions[action])
    obs = next_obs
    return obs, obs_action_pairs

# def execute_1_policy(obs, n, env, obs_action_pairs):
#     if n == 1:
#         policy = [1]
#     else:
#         assert False
#     for action in policy:
#         if isinstance(action, int):
#             obs, obs_action_pairs = execute_transition(obs, action, env, obs_action_pairs)
#         else:
#             obs, obs_action_pairs = action(obs, env, obs_action_pairs)
#     return obs, obs_action_pairs

def execute_n_policy(obs, n, env, obs_action_pairs):
    if n == 1:
        policy = [1]
    elif n == 2:
        policy = [0, 1, 2]
    elif n == 3:
        policy = [1, 0, 5, 1, 3, 2, 1]
    elif n == 4:
        policy = [0, 1, 2, 0, 4, 5, 0, 1, 2, 3, 4, 2, 0, 1, 2]
    else:   
        assert False
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
        print('{}\nn = {}\n{}'.format('*'*40, n, '*'*40))
        obs_action_pairs = dict()
        env = MyHanoiGym(n=n)
        obs = env.reset()
        print('Obs: {}'.format(obs))
        print('State: {}\n{}'.format(env.get_state(), env.text_render()))
        obs, obs_action_pairs = execute_n_policy(obs, n, env, obs_action_pairs)
        print('obs_action_pairs')
        pprint.pprint(obs_action_pairs)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1, help='number of disks')
    args = parser.parse_args()
    test_hanoi_gym_n(max_n=args.n)
