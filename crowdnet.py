from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, AveragePooling2D, Merge, BatchNormalization
from keras.models import Sequential
from preprocessing import load_data, preprocess_data


def create_left_branch(model=Sequential(), kernel_size=(3, 3), input_shape=(225, 225, 1)):
  model.add(Convolution2D(64, *kernel_size, input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Convolution2D(64, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(128, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Convolution2D(128, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(256, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Convolution2D(256, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Convolution2D(256, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Convolution2D(512, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Convolution2D(512, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Convolution2D(512, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.5))

  model.add(Convolution2D(512, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Convolution2D(512, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  return model

def create_right_branch(model=Sequential(), kernel_size=(5, 5),  input_shape=(225, 225, 1)):
  model.add(Convolution2D(24, *kernel_size, input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(5, 5)))

  model.add(Convolution2D(24, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(5, 5)))

  model.add(Convolution2D(24, *kernel_size))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(5, 5)))

  return model

def create_model(model=Sequential(), input_shape=(225, 225, 1)):
  left_branch = create_left_branch(input_shape=input_shape)
  right_branch = create_right_branch(input_shape=input_shape)

  model.add(Merge([left_branch, right_branch], mode='concat'))
  model.add(Convolution2D(1, 1, 1))
  model.add(Activation('relu'))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

if __name__ == '__main__':
  im_generator = load_data()
  model = create_model()
  model.fit_generator(im_generator, nb_epoch=100, validation_split=0.2, verbose=True)
  model.save('weights.h5')
