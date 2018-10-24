from MlModels import Predictor, Autoencoder
from GetThatReplay import gameFromFile, getControls
from keras.models import load_model
import numpy as np
import math


def sampleGameFrames(game, player, start, frames, values):
    for i in range(start, start + frames):
        vals = []
        for name in values:
            v = player.data.get(name).get(i)
            if v == False or v == None or np.isnan(v):
                v = 0
            elif v == True:
                v = 1
            vals.append(v)
        vals += sampleBallFrame(game, i)
        controls = getControls(player, i)
        yield normalize_state(vals), controls


values = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y',
          'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'boost']


def sampleBallFrame(game, frame):
    arr = []
    for name in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']:
        v = game.ball.get(name).get(frame)
        if v == False or v == None or np.isnan(v):
            v = 0
        elif v == True:
            v = 1
        arr.append(v)
    return arr


def normalize_state(state):
    maxVals = [3088, 5424, 1826, math.pi, math.pi, math.pi, 23000, 23000, 23000,
               5500, 5500, 5500, 100, 3088, 5424, 1826, 60000, 60000, 60000, 6000, 6000, 6000]
    maxVals = np.array(maxVals)
    return np.array(state)/maxVals


# 22 state based things
# 8 control based things
load = False  # Set to true to start where you left off
models = []
game = gameFromFile("Replay")  # Replay File path without extension
if not load:
    autoencoder = Autoencoder(latent_size=4)
    predictor = Predictor()
    states = []
    controls = []
    for state, control in sampleGameFrames(game, game.players[1], 0, len(game.frames), values):
        states.append(np.array(state))
        controls.append(np.array(control))
    states = np.array(states)
    controls = np.array(controls)
    autoencoder.fit(states, 100, batch_size=32)
    predictor.fit(states, controls, 100, batch_size=32)
    del states
    del controls
    print("Copying Models...")

    for i in range(15):
        models.append([autoencoder.copy(), predictor.copy()])
    print("Copied Models.")
else:
    for i in range(15):
        ae = Autoencoder(model=load_model("Models\\autoencoder" + str(i) + ".h5"))
        pre = Predictor(model=load_model("Models\\predictor" + str(i) + ".h5"))
        models.append([ae, pre])

frequency = [1]*len(models)
while True:
    try:
        num = 30
        high = len(game.frames) - num
        start = np.random.randint(0, high=high)
        states = []
        controls = []
        for state, control in sampleGameFrames(game, game.players[1], start, num, values):
            states.append(state)
            controls.append(control)
        states = np.array(states)
        controls = np.array(controls)
        evals = []
        for model in models:
            evals.append(model[0].evaluate(states))
        best = np.argmin(evals)
        frequency[best] += 1
        f = sum(frequency) - np.array(frequency)
        f = f / np.sum(f)
        if np.random.randint(0, high=100) < 10:
            best = np.random.choice(len(models), p=f)
            print("Beep, boop: Random Thing.")
        print(frequency)
        models[best][0].fit(states, 1)
        models[best][1].fit(states, controls, 1)

    except KeyboardInterrupt:
        print('Saving models...')
        for i in range(len(models)):
            models[i][0].save("Models\\autoencoder" + str(i) + ".h5")
            models[i][1].save("Models\\predictor" + str(i) + ".h5")
        print("Done")
        break
