from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from MlModels import Autoencoder, Predictor
import numpy as np
import math
# Load Models


class PythonExample(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        print("loading models...")
        self.models = self.load_models(15)
        print("finished loading models")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        state = self.get_state(packet)
        action, network_num = self.choose_action(state)
        self.update_controller_state(action)
        self.render(network_num)
        return self.controller_state

    def render(self, network_num):
        self.renderer.begin_rendering()
        self.renderer.draw_string_2d(10, 10 + 50 * self.index, 5, 5, "Network In Use: " +
                                     str(network_num), self.renderer.white())
        self.renderer.end_rendering()

    def choose_action(self, state):
        # evaluate each AE
        evals = []
        for model in self.models:
            evals.append(model[0].evaluate(state))
        # choose the one that did the best
        num = np.argmin(evals)
        predictor = self.models[num][1]
        # use it to make the output
        action = predictor.predict(state)
        return action, num

    def get_state(self, packet):
        t = 1
        if self.team == 1:
            t *= -1
        ball = packet.game_ball.physics
        me = packet.game_cars[self.index]
        my_physics = me.physics
        my_loc = [my_physics.location.x / 4096,
                  my_physics.location.y / 5120 * t, my_physics.location.z / 2044]
        my_rot = [my_physics.rotation.pitch / math.pi, my_physics.rotation.yaw /
                  math.pi, my_physics.rotation.roll / math.pi]
        my_vel = [my_physics.velocity.x / 2300,
                  my_physics.velocity.y / 2300 * t, my_physics.velocity.z / 2300]
        my_ang_vel = [my_physics.angular_velocity.x / 5.5,
                      my_physics.angular_velocity.y * t / 5.5, my_physics.angular_velocity.z / 5.5]
        my_boost = [me.boost / 100]
        ball_loc = [ball.location.x / 4096, ball.location.y * t / 5120, ball.location.z / 2044]
        ball_vel = [ball.velocity.x / 6000, ball.velocity.y * t / 6000, ball.velocity.z / 6000]
        ball_ang_vel = [ball.angular_velocity.x / 6,
                        ball.angular_velocity.y * t / 6, ball.angular_velocity.z / 6]
        state = np.array([my_loc + my_rot + my_vel + my_ang_vel + my_boost +
                          ball_loc + ball_vel + ball_ang_vel])
        return state

    def update_controller_state(self, arr):
        arr = arr[0]
        self.controller_state.throttle = arr[0]
        self.controller_state.steer = arr[1]
        self.controller_state.pitch = arr[2]
        self.controller_state.yaw = arr[3]
        self.controller_state.roll = arr[4]
        self.controller_state.jump = True if arr[5] > 0.5 else False
        self.controller_state.boost = True if arr[6] > 0.5 else False
        self.controller_state.handbrake = True if arr[7] > 0.5 else False

    def load_models(self, num):
        from keras.models import load_model
        models = []
        for i in range(num):
            ae = Autoencoder(model=load_model(
                "Models\\autoencoder" + str(i) + ".h5"))
            pre = Predictor(model=load_model(
                "Models\\predictor" + str(i) + ".h5"))
            models.append([ae, pre])
        return models
