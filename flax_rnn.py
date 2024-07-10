import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np
import matplotlib.pyplot as plt
import optax
from functools import partial

'''
NOTES:
    - Time dimension expected after batch dimension (Unless time_major=True)
'''

#
# rng = random.PRNGKey(0)
# x = jnp.ones((10, 50, 32))  # batch, time, features
# lstm = nn.RNN(nn.LSTMCell(64))
#
# rng, key = random.split(rng)
# variables = lstm.init(key, x)
# y = lstm.apply(variables, x)  # (10, 50, 64)


class LSTM(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        ScanLSTM = nn.scan(
                nn.OptimizedLSTMCell, variable_broadcast="params",
                split_rngs={"params": False}, in_axes=1, out_axes=1)

        lstm = ScanLSTM(self.features)
        input_shape = x[:, 0].shape
        carry = lstm.initialize_carry(random.key(0), input_shape)
        carry, x = lstm(carry, x)
        x = nn.Dense(1)(x)
        return x

@jax.jit
def update(lstm_state, x_batch, y_batch):
    def loss_fn(params):
        predictions = lstm_state.apply_fn(params, x_batch)
        return jnp.mean((predictions - y_batch)**2)

    loss, grads = jax.value_and_grad(loss_fn)(lstm_state.params)
    lstm_state = lstm_state.apply_gradients(grads=grads)
    return lstm_state, loss


#@partial(jax.jit, static_argnames=["x_range", "batch_size"])
def generate_batch(key: jnp.ndarray, x_range: float, batch_size: int):
    phase = random.randint(key, shape=(batch_size,), minval=0, maxval=x_range)
    # Get x_range for given phase
    gen_seq = jax.vmap(lambda phase: jnp.linspace(phase, x_range+phase, num=time_window))
    gen_label = jax.vmap(lambda phase: jnp.sin(jnp.linspace(phase, x_range+phase, num=time_window)))

    x = jnp.expand_dims(gen_seq(phase), -1)
    y = jnp.expand_dims(gen_label(phase), -1)
    # plt.plot( gen_seq(jnp.array([1])).flatten(), gen_label(jnp.array([1])).flatten() )
    # plt.show()

    return x, y



def create_train_state(rng):
    x_batch, _ = generate_batch(rng, x_range, batch_size)
    model = LSTM(features=128)
    params = model.init(rng, x_batch)
    tx = optax.adam(1e-3)
    # jit the apply fn
    return TrainState.create(params=params, apply_fn=jax.jit(model.apply), tx=tx)


# @partial(jax.jit, static_argnames=["x_range", "batch_size"])
def train(lstm_state, key, x_range, batch_size, epochs):
    losses = []
    for epoch in range(epochs):
        rng, key = random.split(key)
        x_batch, y_batch = generate_batch(key, x_range, batch_size)
        lstm_state, loss = update(lstm_state, x_batch, y_batch)
        # print(f"Loss: {loss}")
        losses.append(loss)

        if epoch % 1000 == 0:
            print(epoch, np.mean(losses[:-1000]))
            # print(losses)
    # plt.plot(losses)
    # plt.show()
    return lstm_state


def test(lstm_state, key, x_range, batch_size):
    x_test, y_test = generate_batch(key, x_range, batch_size)
    x_test = x_test[0]
    y_test = y_test[0]

    pred = lstm_state.apply_fn(lstm_state.params, jnp.expand_dims(x_test, 0))[0] # expand and remove dims
    print(f"loss: {jnp.mean((pred-y_test)**2)}")
    print(x_test.shape, pred.shape, y_test.shape)
    plt.plot(x_test.flatten(), y_test.flatten(), label="label")
    plt.plot(x_test.flatten(), pred.flatten(), label="prediction")
    plt.legend()
    plt.show()


rng = random.PRNGKey(0)
batch_size = 64
time_window = 20
x_range = 2*jnp.pi
epochs = 10_000

rng, key = random.split(rng)

lstm_state = create_train_state(key)
lstm_state = train(lstm_state, key, x_range, batch_size, epochs)
rng, key = random.split(rng)
test(lstm_state, key, x_range, batch_size)  # Make me :)
