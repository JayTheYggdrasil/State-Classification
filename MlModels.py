import numpy as np


class Predictor:
    def __init__(self, model=None):
        if model != None:
            self.model = model
        else:
            print("making model")
            self.make_model()
            print("finalizing model")
            self.finalize_model()
            print("done")

    def make_model(self, input_size=22, output_size=8):
        from keras.models import Model
        from keras.layers import Dense, Dropout, Input
        inputs = Input(shape=(input_size, ))
        hidden = Dropout(0.1)(inputs)
        hidden = Dense(64, activation="relu")(hidden)
        hidden = Dropout(0.1)(hidden)
        hidden = Dense(64, activation="relu")(hidden)
        output = Dense(output_size, activation="tanh")(hidden)
        self.model = Model(inputs=inputs, outputs=output)

    def finalize_model(self):
        optimizer = "rmsprop"
        loss = "mean_squared_error"
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def fit(self, state, controls, epochs, batch_size=1):
        self.model.fit(state, controls, epochs=epochs, batch_size=batch_size)

    def predict(self, state):
        return self.model.predict(state)

    def copy(self):
        from keras.models import clone_model
        copy = Predictor(model=clone_model(self.model))
        copy.finalize_model()
        return copy

    def save(self, filepath):
        self.model.save(filepath)


class Autoencoder:
    def __init__(self, latent_size=None, model=None):
        if model != None:
            self.model = model
        elif latent_size != None:
            print("making model")
            self.make_model(latent_size)
            print("finalizing model")
            self.finalize_model()
            print("done")
        else:
            raise Exception("Model and latent_size cannot both be None")

    def make_model(self, latent_size, state_size=22):
        from keras.models import Model
        from keras.layers import Dense, Dropout, Input
        self.latent_size = latent_size
        input = Input(shape=(state_size, ))
        encoder = Dense(32, activation="relu")(input)
        encoder = Dense(16, activation="relu")(encoder)
        encoder = Dense(latent_size, activation="relu")(encoder)
        decoder = Dense(16, activation="relu")(encoder)
        decoder = Dense(32, activation="relu")(decoder)
        output = Dense(state_size, activation="tanh")(decoder)
        self.model = Model(inputs=input, outputs=output)

    def finalize_model(self):
        optimizer = "rmsprop"
        loss = "mean_squared_error"
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, state, epochs, batch_size=1):
        self.model.fit(state, state, epochs=epochs, batch_size=batch_size)

    def evaluate(self, state):
        return self.model.evaluate(x=state, y=state, verbose=0)

    def copy(self):
        from keras.models import clone_model
        copy = Autoencoder(model=clone_model(self.model))
        copy.finalize_model()
        return copy

    def save(self, filepath):
        self.model.save(filepath)
