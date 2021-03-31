import os, logging

# only print error messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import random
from scipy.stats import norm, lognorm


class Agent:
    """
    Agent wrapper:
        @args
            model: tf.keras.Model (or simply callable bit than some functionalities are not supported) returning dictionary with the possible keys: 'q_values' or 'policy' or 'mus' and 'sigmas' for continuous policies, optional 'value_estimate', containing tensors
            weights: None or weights of the model
            action_sampling_type: string, supported are 'thompson', 'epsilon_greedy', 'discrete_policy' and 'continuous_normal_diagonal'
            epsilon: float for epsilon greedy sampling
            temperature: float for thompson sampling
            value_estimate: boolean if agent returns value estimate
            model_kwargs: dict, optional model initialization specifications
            is_tf: boolean if model init is needed
            test: boolean, if testing (only greedy sampling)
    """

    def __init__(
        self,
        model,
        weights=None,
        action_sampling_type="thompson",
        temperature=1,
        epsilon=0.95,
        value_estimate=False,
        input_shape=None,
        model_kwargs={},
        is_tf=True,
        test=False,
    ):

        super(Agent, self).__init__
        # logging.basicConfig(
        #    filename=f"logging/agent.log", level=logging.DEBUG
        # )
        self.model_kwargs = model_kwargs
        self.model = model(**self.model_kwargs)

        self.weights = weights
        if is_tf:
            self.initialize_model(self.model, input_shape)
        self.model.set_weights(weights)

        self.action_sampling_type = action_sampling_type
        self.temperature = temperature
        self.epsilon = epsilon
        self.value_estimate = value_estimate
        self.test = test

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def initialize_model(self, model, input_dummy):

        if hasattr(model, "tensorflow"):
            assert (
                input_dummy != None
            ), 'You have a tensorflow model with no input shape specified for weight initialization. \n Specify input_shape in "model_kwargs" or specify as False if not needed'
        model(input_dummy)

    # agent readout handler
    def act_experience(self, state, return_log_prob=False):
        output = {}
        # creating network dict
        network_out = self.model(state)

        if self.test:
            old_e = self.epsilon
            self.epsilon = 0
            old_t = self.temperature
            self.temperature = 0.000001

        if self.action_sampling_type == "epsilon_greedy":
            logits = network_out["q_values"]
            if tf.is_tensor(logits):
                logits = logits.numpy()
            if random.random() > self.epsilon:
                action = np.argmax(logits, axis=-1)
                # log prob of epsilon
                if return_log_prob:
                    output["log_probability"] = np.asarray(
                        [np.log(self.epsilon + 0.00000001)] * logits.shape[0],
                        dtype=np.float32,
                    )
            else:
                # log prob of 1-epsilon
                action = [
                    random.randrange(logits.shape[-1]) for _ in range(logits.shape[0])
                ]
                action = np.asarray(action)
                if return_log_prob:
                    output["log_probability"] = np.asarray(
                        [np.log(1 - self.epsilon + 0.00000001)] * logits.shape[0],
                        dtype=np.float32,
                    )
            output["action"] = action

        elif self.action_sampling_type == "thompson":
            # q values
            logits = network_out["q_values"]
            logits = tf.nn.softmax(logits)
            action = tf.squeeze(
                tf.random.categorical(logits / self.temperature, 1)
            ).numpy()
            output["action"] = action
            action = action.tolist()
            if tf.is_tensor(logits):
                logits = logits.numpy()
            if return_log_prob:
                if logits.shape[0] > 1:
                    output["log_probability"] = np.log(
                        [logits[i][a] for i, a in zip(range(logits.shape[0]), action)]
                    )
                else:
                    output["log_probability"] = np.log([logits[0][action]])

        elif self.action_sampling_type == "continuous_normal_diagonal":

            mus, sigmas = network_out["mu"].numpy(), network_out["sigma"].numpy()
            action = norm.rvs(mus, sigmas)
            output["action"] = action

            if return_log_prob:
                output["log_probability"] = np.sum(norm.logpdf(action, mus, sigmas))

        elif self.action_sampling_type == "discrete_policy":
            logits = network_out["policy"]
            if self.test:
                action = np.argmax(logits, axis=-1)
            else:
                action = tf.squeeze(tf.random.categorical(logits, 1)).numpy()
            output["action"] = action
            action = action.tolist()
            if tf.is_tensor(logits):
                logits = logits.numpy()
            if return_log_prob:
                if logits.shape[0] > 1:
                    output["log_probability"] = np.log(
                        [logits[i][a] for i, a in zip(range(logits.shape[0]), action)]
                    )
                else:
                    output["log_probability"] = np.log([logits[0][action]])

        else:
            # logging.warning(f'unsupported sampling method {self.actin_sampling_type}')
            raise NotImplemented

        # pass on value estimate if there
        if self.value_estimate:
            output["value_estimate"] = network_out["value_estimate"]

        if self.test:
            self.epsilon = old_e
            self.temperature = old_t

        return output

    def act(self, state):
        net_out = self.act_experience(state)
        return net_out["action"]

    def max_q(self, x):
        # computes the maximum q-value along each batch dimension
        model_out = self.model(x)
        x = tf.reduce_max(model_out["q_values"], axis=-1)
        return x

    def q_val(self, x, actions):
        model_out = self.model(x)
        q_values = model_out["q_values"]
        x = tf.gather(q_values, actions, batch_dims=1)
        x = tf.expand_dims(x, axis=-1)
        return x

    def v(self, x):
        model_out = self.model(x)
        v = model_out["value_estimate"]
        return v

    def flowing_log_prob(self, state, action, return_entropy=False):
        action = tf.cast(action, dtype=tf.float32)
        network_out = self.model(state)

        if self.action_sampling_type == "continuous_normal_diagonal":
            mus, sigmas = network_out["mu"], network_out["sigma"]
            dist = tf.compat.v1.distributions.Normal(mus, sigmas)
            log_prob = dist.log_prob(action)
            if return_entropy:
                firs_step = tf.constant(np.exp(1), dtype=tf.float32) * (
                    tf.square(sigmas)
                )
                second_step = tf.constant(0.5, dtype=tf.float32) * tf.math.log(
                    2 * tf.constant(np.pi, dtype=tf.float32)
                )
                entropy = firs_step * second_step
                return log_prob, entropy
            return log_prob

        elif self.action_sampling_type == "thompson":
            logits = network_out["q_values"]
            logits = tf.nn.softmax(logits)
            action = tf.cast(action, dtype=tf.int64).numpy().tolist()
            if logits.shape[0] > 1:
                log_prob = tf.math.log(
                    [logits[i][a] for i, a in zip(range(logits.shape[0]), action)]
                )
            else:
                log_prob = tf.math.log([logits[0][action]])
            log_prob = tf.expand_dims(log_prob, -1)
            if return_entropy:
                entropy = -tf.reduce_sum(logits * tf.math.log(logits), axis=-1)
                entropy = tf.expand_dims(entropy, -1)
                return log_prob, entropy
            return log_prob

        elif self.action_sampling_type == "discrete_policy":
            logits = network_out["policy"]

            action = tf.cast(action, dtype=tf.int64).numpy().tolist()
            if logits.shape[0] > 1:
                log_prob = tf.math.log(
                    [logits[i][a] for i, a in zip(range(logits.shape[0]), action)]
                )
            else:
                log_prob = tf.math.log([logits[0][action]])
            log_prob = tf.expand_dims(log_prob, -1)
            if return_entropy:
                entropy = -tf.reduce_sum(logits * tf.math.log(logits), axis=-1)
                entropy = tf.expand_dims(entropy, -1)
                return log_prob, entropy
            else:
                return log_prob

        elif self.action_sampling_type == "epsilon_greedy":
            logits = network_out["q_values"]
            if tf.is_tensor(logits):
                logits = logits.numpy()
            log_prob = np.asarray([np.log(self.epsilon + 0.00000001)] * logits.shape[0])
            return tf.cast(tf.expand_dims(log_prob, -1), dtype=tf.float32)

        else:
            print(
                f"flowing log probabilities not yet implemented for sampling type {self.action_sampling_type}"
            )
            raise NotImplemented
