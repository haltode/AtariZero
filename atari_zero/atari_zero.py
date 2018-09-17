import gym
import numpy as np
import skimage.color
import skimage.transform
import keras


def preprocess_frame(frame):
    gray_frame = skimage.color.rgb2gray(frame)
    resized_frame = skimage.transform.resize(gray_frame, [84, 84])
    # Store on smallest type possible (uint8) because replay experience will
    # take huge amount of memory, hence using float is very costly
    compact_frame = np.uint8(resized_frame * 255)
    return compact_frame


def fit_single_batch(model, gamma,
                     start_states, actions, rewards, next_states, is_terminal):
    # Predict Q values of the next states
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    next_Q_values[is_terminal] = 0

    # Calculate Q values of start states
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)

    # Fit Keras model
    training_data = [start_states, actions]
    target_data = actions * Q_values[:, None]
    model.fit(
        training_data, target_data,
        nb_epoch=1, batch_size=len(start_states), verbose=0
    )


def atari_model():
    ATARI_SHAPE = (84, 84, 4)
    NB_ACTIONS = 3

    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((NB_ACTIONS, ), name='mask')

    normalized = keras.layers.Lambda(lambda x: x / 255.)(frames_input)

    layer1 = keras.layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu'
    )(normalized)
    layer2 = keras.layers.convolutional.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu'
    )(layer1)
    flat_layer2 = keras.layers.core.Flatten()(layer2)
    layer3 = keras.layers.Dense(256, activation='relu')(flat_layer2)

    output = keras.layers.Dense(NB_ACTIONS)(layer3)
    filtered_output = keras.layers.Multiply(name="Q_values")(
        [output, actions_input]
    )

    model = keras.models.Model(
        inputs=[frames_input, actions_input], outputs=filtered_output
    )
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')
    return model


env = gym.make('BreakoutDeterministic-v4')
env.reset()
env.render()

is_done = False
while not is_done:
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    preprocess_frame(frame)
    atari_model()
    env.render()
