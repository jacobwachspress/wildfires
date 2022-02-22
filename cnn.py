import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Lambda, BatchNormalization
import tensorflow as tf
import numpy as np
import json
import os
import sys


def main(date, config_number, path='data/input/configs'):

    config_file = f'{path}/{date}/{config_number}/config.json'

    with open(config_file, 'r') as json_file:
        config = json.load(json_file)

    if not os.path.isdir(config['input_path']):
        generate_input_numpy_data(config)

    model = make_model(config)

    all_IDs = np.load(f"{config['input_path']}/IDs.npy")
    train, val, train_inc_ids, val_inc_ids = train_val_split(all_IDs)
    	
    fit_model(config, model, train, val)

    return 1


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
    dist[i, j] = 1

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


def basic_cnn_preprocess(path, inc_id, fire_level_info, day, i, j, dim, time, fxx,
                         threshold, layer_names):

    slope = fire_level_info['slope']
    aspect = fire_level_info['aspect']
    fire = fire_level_info['fire'][day-1, :, :]
    past_burn = fire_level_info['past_burn'][day-1, :, :]
    next_fire = fire_level_info['fire'][day, :, :]

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

    layers['fire'] = fire
    layers['past_burn'] = past_burn

    to_stack = [layers[name] for name in layer_names]

    stack = np.dstack(to_stack)
    stack = stack[i - dim:i + dim + 1, j - dim:j + dim + 1, :]

    output = (next_fire[i, j] > threshold).astype(int)

    return stack, output


def prior_burn(arr):

    # get the index of the last time there was no burn
    last_no_burn = np.where(arr == 0)[0]

    # if there was a time with no burn
    if len(last_no_burn) > 0:
        last_no_burn = last_no_burn[-1]
        prev_burn_days = np.sum(arr[:last_no_burn])
    else:
        prev_burn_days = 0

    return prev_burn_days


def get_fire_level_info(path, inc_ids, threshold):
    fire_level_info = {}

    for inc_id in inc_ids:
        fire_level_info[inc_id] = {}

        days = sorted([int(i) for i in os.listdir(f'{path}/{inc_id}') if '.' not in i])
        fire_days = [np.load(f'{path}/{inc_id}/{day}/fire.npy') for day in days]
        fire = np.asarray(fire_days)
        fire_level_info[inc_id]['fire'] = fire

        burned = (fire > threshold).astype(int)
        past_burn = np.asarray([np.apply_along_axis(prior_burn, 0, burned[:day, :, :])
                                for day in range(burned.shape[0])])
        fire_level_info[inc_id]['past_burn'] = past_burn

        with open(f'{path}/{inc_id}/static_data.json', 'r') as f:
            static_data_lookup = json.load(f)

        static_data = np.load(f'{path}/{inc_id}/static_data.npy')
        slope = static_data[:, :, static_data_lookup['slope']].clip(0, np.inf)
        aspect = static_data[:, :, static_data_lookup['aspect']].clip(0, np.inf)

        fire_level_info[inc_id]['slope'] = slope
        fire_level_info[inc_id]['aspect'] = aspect

    return fire_level_info


def train_val_split(IDs, ideal_train_frac=0.8, eps=0.05):
    # get all inc_ids
    inc_ids = IDs[:, 0]

    # get the observation counts for each incident
    incs, counts = np.unique(inc_ids, return_counts=True)

    ## assign indices for train and validation set so that each fire is completely in one set ##

    # initialize fraction in validation set to unacceptable value
    val_frac = -1

    # random seed for reproducibility
    seed = 8

    # while fraction in validation set is too far off from ideal
    while abs(val_frac + ideal_train_frac - 1) > eps:

        # set seed
        np.random.seed(seed)
        seed += 1

        # shuffle the IDs
        order = np.arange(len(incs))
        np.random.shuffle(order)
        incs = incs[order]
        counts = counts[order]

        # find the first index where we have enough training samples (might be too many)
        split = np.where(np.cumsum(counts) / np.sum(counts) > ideal_train_frac - eps)[0][0] + 1

        # grab the corresponding train and validation inc_ids
        train_inc_ids = incs[:split]
        val_inc_ids = incs[split:]

        # grab the indices from IDs corresponding to train and validation pixels
        train = np.where(np.isin(IDs[:, 0], train_inc_ids))[0]
        val = np.where(np.isin(IDs[:, 0], val_inc_ids))[0]

        # check that the lengths make sense
        if len(train) + len(val) != len(IDs):
            raise ValueError('Some input observations not placed into exactly one of train and validation sets')
        # update validation fraction
        val_frac = len(val) / len(IDs)

    return train, val, train_inc_ids, val_inc_ids

def generate_input_numpy_data(config):

    in_path = config['processed_path']
    out_path = config['input_path']
    layers = config['layers']
    dim = config['dim']
    buffer = config['buffer']
    threshold = config['threshold']
    max_samples = config['max_samples']

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    assert buffer >= dim

    time = '12z'
    fxx = 'f00'

    all_IDs = []
    inc_ids = os.listdir(in_path)

    for inc_id in inc_ids:
        shape = np.load(f'{in_path}/{inc_id}/1/fire.npy').shape[0:2]
        days = sorted([int(i) for i in os.listdir(f'{in_path}/{inc_id}') if '.' not in i])
        days = days[:-1]
        IDs = [(inc_id, day, i, j) for day in days for i in range(buffer, shape[0] - buffer + 1) \
               for j in range(buffer, shape[1] - buffer + 1)]
        all_IDs = all_IDs + IDs

    all_IDs = np.asarray(all_IDs)

    l = len(all_IDs)
    if l > max_samples:

        eps = max_samples / 10 / len(all_IDs)
        ixs, _, inc_ids, _ = train_val_split(all_IDs, ideal_train_frac=max_samples/len(all_IDs),
                                                 eps=eps)
        all_IDs = all_IDs[ixs]

        print(f'Cut {l} samples down to {len(all_IDs)}, {len(inc_ids)} fires kept')

    fire_level_info_dict = get_fire_level_info(in_path, inc_ids, threshold)

    np.save(f'{out_path}/IDs.npy', np.asarray(all_IDs))
    outps = []
    for ix, ID in enumerate(all_IDs):
        inc_id, day, i, j = ID
        day = int(day)
        i = int(i)
        j = int(j)
        fire_level_info = fire_level_info_dict[inc_id]
        inp, outp = basic_cnn_preprocess(in_path, inc_id, fire_level_info,
                                         day, i, j, dim, time, fxx, threshold, layers)

        if np.any(inp != inp):
            print(ID)
            raise ValueError('NaNs')

        np.save(f'{out_path}/{ix}.npy', inp)
        outps.append(outp)
    np.save(f'{out_path}/outputs.npy', np.asarray(outps))


    return 1


def make_model(config):

    conv_ixs = [i for i, layer in enumerate(config['layers']) if layer in config['conv_layers']]
    point_ixs = [i for i, layer in enumerate(config['layers']) if layer in config['point_layers']]
    input_shape = np.load(f'{config["input_path"]}/0.npy').shape

    input_data = keras.Input(shape=input_shape)

    m = int((input_shape[0] + 1) / 2)

    x = Lambda(lambda t: tf.stack([t[..., i] for i in conv_ixs], axis=-1), name='Get_layers_for_conv')(input_data)
    x = BatchNormalization()(x)
    if config['model']['conv_filters'] is not None:
        x = Conv2D(config['model']['conv_filters'], kernel_size=(2, 2), activation="relu")(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    if config['model']['conv_dropout'] is not None:
        x = Dropout(config['model']['conv_dropout'])(x)
    y = Lambda(lambda t: tf.stack([t[..., i] for i in point_ixs], axis=-1), name='Get_layers_for_point_estimates')(
        input_data)
    y = Lambda(lambda t: t[:, m, m], name='Get_point_data')(y)
    y = BatchNormalization()(y)
    y = keras.layers.Concatenate()([x, y])
    if config['model']['dense_nodes'] is not None:
        y = Dense(config['model']['dense_nodes'], activation="relu")(y)
    if config['model']['dense_dropout'] is not None:
        y = Dropout(config['model']['dense_dropout'])(y)

    output_data = Dense(2, activation="softmax", name="Output")(y)

    return keras.Model(input_data, output_data)


def fit_model(config, model, train_ix, val_ix):

    out_path = config['out_path']

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    data_path = config['input_path']

    model.compile(**config['compile'])
    if config['learning_rate'] is not None:
        keras.backend.set_value(model.optimizer.learning_rate, config['learning_rate'])

    callbacks = [keras.callbacks.ModelCheckpoint(**config['callbacks'])]
    
    d = config['fit_generator']['class_weight']
    if d is not None:
        d2 = {}
        for key in d:
            d2[int(key)] = d[key]
        config['fit_generator']['class_weight'] = d2

    if config['fit_style'] == 'fit_generator':
        training_generator = DataGenerator(data_path, train_ix)
        validation_generator = DataGenerator(data_path, val_ix)
        model.fit(training_generator,
                  validation_data=validation_generator,
                  callbacks=callbacks,
                  **config['fit_generator'])

    return model


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path, list_IDs, batch_size=32, shuffle=True):
        'Initialization'
        self.dim = np.load(f'{data_path}/1.npy').shape[:2]
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

if __name__ == "__main__":

    date = sys.argv[1]
    config_number = sys.argv[2]
    main(date, config_number)

