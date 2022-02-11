from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Lambda, BatchNormalization
import tensorflow as tf
import numpy as np
import json
from glob import glob
import os


def main():

    layers = ['fire', 'downwind', 'upslope', 'wind', 'TMP:2 m', 'SPFH:2 m', 'LTNG', 'PRATE', 'PRES:surface']
    conv_ixs = [0, 1, 2, 3]
    point_ixs = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    in_path = 'data/test'
    input_path = 'data/cnn_input5'

    cnn.generate_input_numpy_data(in_path, input_path, layers, dim=10, threshold=0.2)

    input_shape = np.load(f'{input_path}/0.npy').shape

    model = cnn.make_model(conv_ixs, point_ixs, input_shape, 'test')

    outputs = np.load(f'{input_path}/outputs.npy')
    ix_to_keep = np.arange(outputs.shape[0])
    np.random.seed(8)
    np.random.shuffle(ix_to_keep)
    split = int(np.floor(0.8 * len(ix_to_keep)))
    train = ix_to_keep[:split]
    val = ix_to_keep[split:]

    model = cnn.fit_model(input_path, model, train, val, out_file='cnn_from_scratch.keras', epochs=2)


def slope_aspect_wind_transform(slope, aspect, u_wind, v_wind, i, j):

    # sometimes the slope or aspect DataFrames do not read in correctly
    if np.any(aspect == -9999) or np.any(slope == -9999):
        raise ValueError('Slope or aspect not read correctly')

    # get the shape of the arrays, make sure they are always the same
    h, w = slope.shape
    for arr in [aspect, u_wind, v_wind]:
        if arr.shape != slope.shape:
            raise ValueError("Arrays not all of same shape")

    # get the distance E and N needed to travel from each pixel to (i, j) pixel
    u_dist = np.zeros((h, w))
    v_dist = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            u_dist[y, x] = j - x
            v_dist[y, x] = y - i

    # get the pixel distance from each pixel to (i, j)
    dist = (v_dist ** 2 + u_dist ** 2) ** 0.5

    # get the U and V component of the direction of the upslope
    # (270 degrees is up E, 180 is up N)
    u_upslope = - np.sin(aspect / 180 * np.pi)
    v_upslope = - np.cos(aspect / 180 * np.pi)

    # dot product the upslope vector with the vector from pixel to (i, j)
    upslope = (u_dist * u_upslope  + v_dist * v_upslope) / dist

    # manually set 0/0 value at pixel (i, j) to 1
    upslope[i, j] = 1

    # dot product the downwind vector with the vector from pixel to (i, j)
    wind = (u_wind ** 2 + v_wind ** 2) ** 0.5
    downwind = (u_dist * u_wind + v_dist * v_wind) / (dist * wind)
    downwind[i, j] = 1

    # manually set 0/0 value at pixel (i, j) to 1
    return dist, upslope, downwind, wind


def basic_cnn_preprocess(path, inc_id, day, i, j, dim, time, fxx, threshold, layer_names):
    with open(f'{path}/{inc_id}/static_data.json', 'r') as f:
        static_data_lookup = json.load(f)

    static_data = np.load(f'{path}/{inc_id}/static_data.npy')
    slope = static_data[:, :, static_data_lookup['slope']].clip(0, np.inf)
    aspect = static_data[:, :, static_data_lookup['aspect']].clip(0, np.inf)

    with open(f'{path}/{inc_id}/{day}/weather_{time}.json', 'r') as f:
        weather_data_lookup = json.load(f)
    weather_data = np.load(f'{path}/{inc_id}/{day}/weather_{time}_{fxx}.npy')

    layers = {}
    for key in weather_data_lookup:
        layers[key] = weather_data[:, :, weather_data_lookup[key]]

    u_wind = layers['UGRD:10 m']
    v_wind = layers['VGRD:10 m']

    layers['dist'], layers['upslope'], layers['downwind'], layers['wind'] = \
        slope_aspect_wind_transform(slope, aspect, u_wind, v_wind, i, j)

    layers['inv_dist'] = 1 / layers['dist']
    layers['inv_dist'][i, j] = layers['inv_dist'].min()

    layers['fire'] = np.load(f'{path}/{inc_id}/{day}/fire.npy')

    to_stack = [layers[name] for name in layer_names]

    stack = np.dstack(to_stack)
    stack = stack[i - dim:i + dim + 1, j - dim:j + dim + 1, :]

    next_fire_array = np.load(f'{path}/{inc_id}/{day + 1}/fire.npy')
    output = (next_fire_array[i, j] > threshold).astype(int)

    return stack, output

def generate_input_numpy_data(in_path, out_path, layers, dim=10, threshold=0.2):


    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    time = '12z'
    fxx = 'f00'

    all_IDs = []
    inc_ids = os.listdir(in_path)

    for inc_id in inc_ids:
        shape = np.load(f'{in_path}/{inc_id}/1/fire.npy').shape[0:2]
        days = sorted([int(i) for i in os.listdir(f'{in_path}/{inc_id}') if '.' not in i])
        days = days[:-1]
        IDs = [(inc_id, day, i, j) for day in days for i in range(dim, shape[0] - dim) \
               for j in range(dim, shape[1] - dim)]
        all_IDs = all_IDs + IDs

    outps = []
    for ix, ID in enumerate(all_IDs):
        inc_id, day, i, j = ID
        inp, outp = basic_cnn_preprocess(in_path, inc_id, day, i, j, dim, time, fxx, threshold, layers)

        np.save(f'{out_path}/{ix}.npy', inp)
        outps.append(outp)
    np.save(f'{out_path}/outputs.npy', np.asarray(outps))
    np.save(f'{out_path}/IDs.npy', np.asarray(all_IDs))

    return 1


def make_model(conv_ixs, point_ixs, input_shape, name):
    input_data = keras.Input(shape=input_shape)

    m = int((input_shape[0] + 1) / 2)

    x = Lambda(lambda t: tf.stack([t[..., i] for i in conv_ixs], axis=-1), name='Get_layers_for_conv')(input_data)
    x = BatchNormalization()(x)
    x = Conv2D(5, kernel_size=(2, 2), activation="relu")(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    y = Lambda(lambda t: tf.stack([t[..., i] for i in point_ixs], axis=-1), name='Get_layers_for_point_estimates')(
        input_data)
    y = Lambda(lambda t: t[:, m, m], name='Get_point_data')(y)
    y = BatchNormalization()(y)
    y = keras.layers.Concatenate()([x, y])
    y = Dense(256, activation="relu")(y)
    y = Dropout(0.2)(y)
    output_data = Dense(2, activation="softmax", name="Output")(y)

    return keras.Model(input_data, output_data, name=name)


def fit_model(data_path, model, train_ix, val_ix, out_file='cnn_from_scratch.keras', epochs=15):

    training_generator = DataGenerator(data_path, train_ix)
    validation_generator = DataGenerator(data_path, val_ix)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    callbacks = [keras.callbacks.ModelCheckpoint(
        filepath=out_file,
        save_best_only=True,
        monitor='val_acc')]

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        callbacks=callbacks,
                        epochs=epochs,
                        workers=50)

    return model


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path, list_IDs, batch_size=32, dim=10, shuffle=True):
        'Initialization'
        self.dim = (2*dim + 1, 2*dim+1)
        self.data_path = data_path
        self.batch_size = batch_size
        self.labels = np.load(f'{data_path}/outputs.npy')
        self.list_IDs = list_IDs
        self.n_channels = np.load(f'{data_path}/1.npy').shape[-1]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = np.load(f'{self.data_path}/{ID}.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, y