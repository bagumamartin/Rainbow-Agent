import tensorflow as tf
from tensorflow.keras.saving import custom_object_scope

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


# Detect and configure TPU, GPU, or CPU
# try:
#     # Attempt to use a TPU
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.TPUStrategy(tpu)
#     print("Running on TPU")
# except ValueError:
#     # If TPU is not available, fallback to GPU or CPU
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.list_logical_devices('GPU')
#             print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
#             strategy = tf.distribute.MirroredStrategy()
#         except RuntimeError as e:
#             print(e)
#             strategy = tf.distribute.get_strategy()  # Default strategy for single GPU or CPU
#     else:
#         print("Running on CPU")
#         strategy = tf.distribute.get_strategy()  # Default strategy for CPU

from tensorflow import keras
import numpy as np
import datetime
from .utils.memories import ReplayMemory, RNNReplayMemory, MultiStepsBuffer
from .utils.models import ModelBuilder, AdversarialModelAgregator
import dill
import glob
import json
import os
import types


class Rainbow:
    def __init__(self,
            nb_states, 
            nb_actions, 
            gamma, 
            
            replay_capacity, 
            learning_rate, 
            batch_size,
            epsilon_function = lambda episode, step : max(0.001, (1 - 5E-5)** step), 
            # Model buildes
            window = 1, # 1 = Classic , 1> = RNN
            units = [32, 32],
            dropout = 0,
            adversarial = False,
            noisy = False,
            # Double DQN
            tau = 500, 
            # Multi Steps replay
            multi_steps = 1,
            # Distributional
            distributional = False, nb_atoms = 51, v_min= -200, v_max= 200,
            # Prioritized replay
            prioritized_replay = False, prioritized_replay_alpha =0.65, prioritized_replay_beta_function = lambda episode, step : min(1, 0.4 + 0.6*step/50_000),
            # Vectorized envs
            simultaneous_training_env = 1,
            train_every = 1,
            name = "Rainbow",
        ):
        self.name = name
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.gamma =  tf.cast(gamma, dtype= tf.float32)
        self.epsilon_function = epsilon_function if not noisy else lambda episode, step : 0
        self.replay_capacity = replay_capacity
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta_function = prioritized_replay_beta_function
        self.train_every = train_every
        self.multi_steps = multi_steps
        self.simultaneous_training_env = simultaneous_training_env

        self.recurrent = window > 1
        self.window = window

        self.nb_atoms = nb_atoms
        self.v_min = v_min
        self.v_max = v_max
    
        self.distributional = distributional

        # Memory
        self.replay_memory = ReplayMemory(capacity= replay_capacity, nb_states= nb_states, prioritized = prioritized_replay, alpha= prioritized_replay_alpha)
        if self.recurrent: self.replay_memory = RNNReplayMemory(window= window, capacity= replay_capacity, nb_states= nb_states, prioritized = prioritized_replay, alpha= prioritized_replay_alpha)
        if self.multi_steps > 1: self.multi_steps_buffers = [MultiStepsBuffer(self.multi_steps, self.gamma) for _ in range(simultaneous_training_env)]

        # Models
        # with strategy.scope():
        model_builder = ModelBuilder(
            units = units,
            dropout= dropout,
            nb_states= nb_states,
            nb_actions= nb_actions,
            l2_reg= None,
            window= window,
            distributional= distributional, nb_atoms= nb_atoms,
            adversarial= adversarial,
            noisy = noisy
        )
        input_shape = (None, nb_states)
        self.model = model_builder.build_model(trainable= True)
        self.model.build(input_shape)
        self.model.compile(
            # optimizer= tf.keras.optimizers.legacy.Adam(self.learning_rate, epsilon= 1.5E-4)
            optimizer= tf.keras.optimizers.Adam(self.learning_rate, epsilon= 1.5E-4)
        )

        self.target_model = model_builder.build_model(trainable= False)
        self.target_model.build(input_shape)
        self.target_model.set_weights(self.model.get_weights())


        # Initialize Tensorboard
        # self.log_dir = f"logs/{name}_{self.start_time.strftime('%Y_%m_%d-%H_%M_%S')}"
        # self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        # History
        self.steps = 0
        self.episode_count = -1

        self.losses = []
        self.episode_rewards = [[] for _ in range(self.simultaneous_training_env)]
        self.episode_count = [0 for _ in range(self.simultaneous_training_env)]
        self.episode_steps = [0 for _ in range(self.simultaneous_training_env)]

        #INITIALIZE CORE FUNCTIONS
        # Distributional training
        if self.distributional:
            self.delta_z = (v_max - v_min)/(nb_atoms - 1)
            self.zs = tf.constant([v_min + i*self.delta_z for i in range(nb_atoms)], dtype= tf.float32)

        self.start_time = datetime.datetime.now()
        
    def new_episode(self, i_env):
        self.episode_count[i_env] += 1
        self.episode_steps[i_env] = 0
        self.episode_rewards[i_env] = []
        
    
    def store_replay(self, state, action, reward, next_state, done, truncated, i_env = 0):
        # Case where no multi-steps:
        if self.multi_steps == 1:
            self.replay_memory.store(
                state, action, reward, next_state, done
            )
        else:
            self.multi_steps_buffers[i_env].add(state, action, reward, next_state, done)
            if self.multi_steps_buffers[i_env].is_full():
                self.replay_memory.store(
                    *self.multi_steps_buffers[i_env].get_multi_step_replay()
                )
            
        # Store history
        self.episode_rewards[i_env].append(reward)

        if done or truncated:
            self.log(i_env)
            self.new_episode(i_env)
    def store_replays(self, states, actions, rewards, next_states, dones, truncateds):
        for i_env in range(len(actions)):
            self.store_replay(
                states[i_env], actions[i_env], rewards[i_env], next_states[i_env], dones[i_env], truncateds[i_env], i_env = i_env
            )

    
    def train(self):
        # with strategy.scope():
        self.steps += 1
        for i_env in range(self.simultaneous_training_env): self.episode_steps[i_env] += 1
        if self.replay_memory.size() < self.batch_size or self.get_current_epsilon() >= 1:
            return
        
        if self.steps % self.tau == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        if self.steps % self.train_every == 0:
            batch_indexes, states, actions, rewards, states_prime, dones, importance_weights = self.replay_memory.sample(
                self.batch_size,
                self.prioritized_replay_beta_function(sum(self.episode_count), self.steps)
            )

            loss_value, td_errors = self.train_step(states, actions, rewards, states_prime, dones, importance_weights)
            self.replay_memory.update_priority(batch_indexes, td_errors)

            self.losses.append(float(loss_value))

        # Tensorboard
        # with self.train_summary_writer.as_default():
        #     tf.summary.scalar('Step Training Loss', loss_value, step = self.total_stats['training_steps'])

    def log(self, i_env = 0):
        text_print =f"\
↳ Env {i_env} : {self.episode_count[i_env]:03} : {self.steps: 8d}   |   {self.format_time(datetime.datetime.now() - self.start_time)}   |   Epsilon : {self.get_current_epsilon()*100: 4.2f}%   |   Mean Loss (last 10k) : {np.mean(self.losses[-10_000:]):0.4E}   |   Tot. Rewards : {np.sum(self.episode_rewards[i_env]): 8.2f}   |   Rewards (/1000 steps) : {1000 * np.sum(self.episode_rewards[i_env]) / self.episode_steps[i_env]: 8.2f}   |   Length : {self.episode_steps[i_env]: 6.0f}"
        print(text_print)
    
    def get_current_epsilon(self, delta_episode = 0, delta_steps = 0):
        # if self.noisy: return 0
        return self.epsilon_function(sum(self.episode_count) + delta_episode, self.steps + delta_steps)

    def e_greedy_pick_action(self, state):
        epsilon = self.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.nb_actions)
        return self.pick_action(state)

    def e_greedy_pick_actions(self, states):
        epsilon = self.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.nb_actions, size = self.simultaneous_training_env)
        return self.pick_actions(states).numpy()

    def format_time(self, t :datetime.timedelta):
        h = t.total_seconds() // (60*60)
        m  = (t.total_seconds() % (60*60)) // 60
        s = t.total_seconds() % (60)
        return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"
   

    def train_step(self, *args, **kwargs):
        if self.distributional: return self._distributional_train_step(*args, **kwargs)
        return self._classic_train_step(*args, **kwargs)
    def pick_action(self, *args, **kwargs):
        if self.distributional: return int(self._distributional_pick_action(*args, **kwargs))
        return int(self._classic_pick_action(*args, **kwargs))
    def pick_actions(self, *args, **kwargs):
        if self.distributional: return self._distributional_pick_actions(*args, **kwargs)
        return self._classic_pick_actions(*args, **kwargs)

    # Classic DQN Core Functions
    @tf.function
    def _classic_train_step(self, states, actions, rewards_n, states_prime_n, dones_n, weights):
        best_ap =tf.argmax(self.model(states_prime_n, training = False), axis = 1)
        max_q_sp_ap = tf.gather_nd(
            params = self.target_model(states_prime_n, training = False), 
            indices = best_ap[:, tf.newaxis],
            batch_dims=1)
        q_a_target = rewards_n + tf.cast(dones_n, dtype= tf.float32) *(self.gamma ** self.multi_steps) * max_q_sp_ap

        with tf.GradientTape() as tape:
            q_prediction = self.model(states, training = True)
            q_a_prediction = tf.gather_nd(
                params = q_prediction, 
                indices = tf.cast(actions[:, tf.newaxis], dtype= tf.int32),
                batch_dims=1)
            td_errors = tf.math.abs(q_a_target - q_a_prediction)
            loss_value = tf.math.reduce_mean(
                tf.square(td_errors)*tf.cast(weights, tf.float32)
            )
        # self.model.optimizer.minimize(loss_value, self.model.trainable_weights, tape = tape)
        gradients = tape.gradient(loss_value, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss_value, td_errors

    @tf.function
    def _classic_pick_actions(self, states):
        return tf.argmax(self.model(states, training = False), axis = 1)
    
    @tf.function
    def _classic_pick_action(self, state):
        return tf.argmax(self.model(state[tf.newaxis, ...], training = False), axis = 1)

    # Distributional Core Functions
    @tf.function
    def _distributional_pick_actions(self, states):
        return tf.argmax(self._distributional_predict_q_a(self.model,states, training = False), 1)

    @tf.function
    def _distributional_pick_action(self, state):
        return tf.argmax(self._distributional_predict_q_a(self.model,state[tf.newaxis, :], training = False), axis = 1)

    @tf.function
    def _distributional_predict_q_a(self, model, s, training = True):
        p_a = model(s, training = training)
        q_a = tf.reduce_sum(p_a * self.zs, axis = -1)
        return q_a

    @tf.function
    def _distributional_train_step(self, states, actions, rewards, states_prime, dones, weights):
        best_a__sp =tf.argmax(
            self._distributional_predict_q_a(self.model, states_prime, training = False),
            axis = 1)

        Tz = rewards[..., tf.newaxis] * tf.ones(shape=(self.batch_size, self.nb_atoms)) + tf.cast(dones[..., tf.newaxis], dtype = tf.float32) * (self.gamma ** self.multi_steps) * self.zs * tf.ones(shape=(self.batch_size,self.nb_atoms))
        Tz = tf.clip_by_value(Tz, self.v_min, self.v_max)

        b_j = (Tz - self.v_min)/self.delta_z
        l = tf.math.floor(b_j)
        u = tf.math.ceil(b_j)

        p__max_ap_sp = tf.gather_nd(
            params = self.target_model(states_prime, training = False), 
            indices = best_a__sp[:, tf.newaxis],
            batch_dims=1)
        
        m_l = p__max_ap_sp * (u - b_j)
        m_u = p__max_ap_sp * (b_j - l)

        u = tf.cast(u, tf.int32)
        l = tf.cast(l, tf.int32)

        batch_indexes = (tf.range(self.batch_size)[..., tf.newaxis] * tf.ones(shape=(self.batch_size, self.nb_atoms), dtype= tf.int32))[..., tf.newaxis]

        l_ = tf.concat([batch_indexes, l[:, :,tf.newaxis]], axis= -1)
        u_ = tf.concat([batch_indexes, u[:, :,tf.newaxis]], axis= -1)

        m = tf.scatter_nd(l_, m_l[:], shape=(self.batch_size, self.nb_atoms,)) + tf.scatter_nd(u_, m_u[:], shape=(self.batch_size, self.nb_atoms,))

        with tf.GradientTape() as tape:
            p__s_a = tf.gather_nd(
                params = self.model(states, training = True), 
                indices = tf.cast(actions[:, tf.newaxis], dtype= tf.int32),
                batch_dims=1)
            td_errors = - tf.reduce_sum( m * tf.math.log(tf.clip_by_value(p__s_a , 1E-6, 1.0 - 1E-6)), axis = -1)
            td_errors_weighted = td_errors *tf.cast(weights, tf.float32)
            loss_value = tf.math.reduce_mean(td_errors_weighted)
        
        # self.model.optimizer.minimize(loss_value, self.model.trainable_weights, tape = tape)
        gradients = tape.gradient(loss_value, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss_value, td_errors

    def save(self, path, **kwargs):
        self.saved_path = path
        if not os.path.exists(path): os.makedirs(path)

        if self.model is not None: self.model.save(f"{path}/model.keras")
        if self.target_model is not None: self.target_model.save(f"{path}/target_model.keras")
        
        with open(f'{path}/agent.pkl', 'wb') as file:
            dill.dump(self, file)
        for key, element in kwargs.items():
            if isinstance(element, dict):
                with open(f'{path}/{key}.json', 'w') as file:
                    dill.dump(element, file)
            else:
                with open(f'{path}/{key}.pkl', 'wb') as file:
                    dill.dump(element, file)

    def __getstate__(self):
        print("Saving agent ...")
        state = self.__dict__.copy()
        state.pop('model', None)
        state.pop('target_model', None)
        state.pop('replay_memory', None)
        return state


def load_agent(path, retrain=False):
    with open(f'{path}/agent.pkl', 'rb') as file:
        unpickler = dill.Unpickler(file)
        agent = unpickler.load()

    # Initialize replay memory and buffers based on retrain
    if retrain:
        print("Retrain flag detected, Loading agent for retraining...")
        agent.replay_memory = ReplayMemory(capacity=agent.replay_capacity, nb_states=agent.nb_states, prioritized=agent.prioritized_replay, alpha=agent.prioritized_replay_alpha)
        if agent.recurrent:
            agent.replay_memory = RNNReplayMemory(window=agent.window, capacity=agent.replay_capacity, nb_states=agent.nb_states, prioritized=agent.prioritized_replay, alpha=agent.prioritized_replay_alpha)
        if agent.multi_steps > 1:
            agent.multi_steps_buffers = [MultiStepsBuffer(agent.multi_steps, agent.gamma) for _ in range(agent.simultaneous_training_env)]
    else:
        print("Loading agent for inference...")

    custom_objects = {"AdversarialModelAgregator": AdversarialModelAgregator}

    with custom_object_scope(custom_objects):
        agent.model = tf.keras.models.load_model(f'{path}/model.keras', compile=False)
        agent.target_model = tf.keras.models.load_model(f'{path}/target_model.keras', compile=False)

    # Re-compile the model to restore the optimizer state
    if retrain:
        agent.model.compile(optimizer=tf.keras.optimizers.Adam(agent.learning_rate, epsilon=1.5E-4))

    other_elements = {}
    other_pathes = glob.glob(f'{path}/*pkl')
    other_pathes.extend(glob.glob(f'{path}/*json'))
    for element_path in other_pathes:
        name = os.path.split(element_path)[-1].replace(".pkl", "").replace(".json", "")
        if name != "agent":
            with open(element_path, 'rb') as file:
                if ".pkl" in element_path:other_elements[name] = dill.load(file)
                elif ".json" in element_path:other_elements[name] = json.load(file)
            
    return agent, other_elements

