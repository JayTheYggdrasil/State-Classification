import carball
import os
from carball.json_parser.game import Game
from carball.controls.controls import ControlsCreator
import numpy as np
import math


def gameFromFile(replayName, dt=(1/30), n=15):  # Do not include file extension
    # Output dim (Players, ReplayLen, n, 8 )
    _json = carball.decompile_replay(replayName + '.replay',
                                     replayName + '.json',
                                     overwrite=True)

    game = Game()
    game.initialize(loaded_json=_json)

    ControlsCreator().get_controls(game)

    return game


def sampleGameFrames(game, n):
    players = []
    numFrames = len(game.frames.time)
    for player in game.players:
        frames = []
        for i in range(numFrames - n):
            frames.append(sampleFrames(game, player, i, i + n))
        players.append(frames)
    return players


def sampleFrames(game, player, start, end):
    frames = []
    for i in range(start, end):
        frames.append(getControls(player, i))
    return frames


def getControls(player, frame):
    #[ Throttle, Steer, pitch, yaw, roll, jump, boost, handbrake ]
    c = player.controls
    throttle = c.throttle.get(frame)
    steer = c.steer.get(frame)
    pitch = c.pitch.get(frame)
    yaw = c.yaw.get(frame)
    roll = c.roll.get(frame)
    jump = c.jump.get(frame)
    boost = c.boost.get(frame)
    handbrake = c.handbrake.get(frame)
    controls = [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    for c in range(len(controls)):
        control = controls[c]
        if control == False or control == None or np.isnan(control):
            controls[c] = 0
        elif control == True:
            controls[c] = 1
    return controls
