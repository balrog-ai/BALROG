import pickle

import numpy as np
import torch


class RLModel:
    def __init__(self, observation_def, action_space, train=False, training_comm=(None, None)):
        # observation_def -- list (name, tuple (shape, dtype))
        self.observation_def = observation_def
        self.action_space = action_space
        self.train = train
        self.input_queue, self.output_queue = None, None
        if self.train:
            training_comm[0].put(pickle.loads(pickle.dumps(self)))  # HACK
            self.input_queue, self.output_queue = training_comm
        else:
            import self_play
            import games.nethack
            checkpoint = torch.load('/checkpoints/nethack/2021-10-08--16-13-24/model.checkpoint')
            config = games.nethack.MuZeroConfig(rl_model=self)
            self.inference_iterator = self_play.SelfPlayNoRay(checkpoint, lambda *a: None, config, 0) \
                .play_game_generator(0, 0, False, config.opponent, 0)
            assert next(self.inference_iterator) is None
            self.is_first_iteration = True

    # def encode_observation(self, observation):
    #     assert sorted(observation.keys()) == sorted(self.observation_def.keys())
    #     ret = []
    #     for key, (shape, dtype) in self.observation_def:
    #         val = observation[key]
    #         assert val.shape == shape, (val.shape, shape)
    #         ret.append(np.array(list(val.reshape(-1).astype(dtype).tobytes()), dtype=np.uint8))
    #     ret = np.concatenate(ret)
    #     return ret

    def encode_observation(self, observation):
        vals = []
        hw_shape = None
        for key, (shape, dtype) in self.observation_def:
            vals.append(observation[key])
            if hw_shape is not None and len(shape) > 1:
                if len(shape) == 2:
                    assert hw_shape == shape, (hw_shape, shape)
                elif len(shape) == 3:
                    assert hw_shape == shape[1:], (hw_shape, shape)
                else:
                    assert 0, hw_shape
            if len(shape) > 1:
                if len(shape) == 2:
                    hw_shape = shape
                elif len(shape) == 3:
                    hw_shape = shape[1:]
                else:
                    assert 0

        vals = [(
                    val.reshape(val.shape[0], *hw_shape) if len(val.shape) == 3 else
                    val.reshape(1, *val.shape) if len(val.shape) == 2 else
                    val.reshape(val.shape[0], 1, 1).repeat(hw_shape[0], 1).repeat(hw_shape[1], 2)
                ).astype(np.float32) for val in vals]
        return np.concatenate(vals, 0)

    def zero_observation(self):
        ret = {}
        for key, (shape, dtype) in self.observation_def:
            ret[key] = np.zeros(shape=shape, dtype=dtype)
        return ret

    def observation_shape(self):
        return self.encode_observation(self.zero_observation()).shape

    # def decode_observation(self, data):
    #     ret = {}
    #     for key, (shape, dtype) in self.observation_def:
    #         arr = np.zeros(shape=shape, dtype=dtype)
    #         s = len(arr.tobytes())
    #         ret[key] = np.frombuffer(bytes(data[:s]), dtype=np.dtype).reshape(shape)
    #         data = data[s:]
    #     assert len(data) == 0
    #     return ret

    def choose_action(self, agent, observation, legal_actions):
        assert len(legal_actions) > 0
        assert all(map(lambda action: action in self.action_space, legal_actions))
        assert len(legal_actions) > 0
        legal_actions = [self.action_space.index(action) for action in legal_actions]
        if self.train:
            self.input_queue.put((observation, legal_actions, agent.score))
            action_id = self.output_queue.get()
            if action_id is None:
                raise KeyboardInterrupt()
        else:
            action_id = self.inference_iterator.send((self.encode_observation(observation), 0, False, 0, legal_actions))
        assert action_id in legal_actions
        return self.action_space[action_id]
