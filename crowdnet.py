from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Merge, BatchNormalization
from keras.models import Sequential
from preprocessing import load_data, preprocess_data


def create_left_branch(model=Sequential(), kernel_size=(3, 3), input_shape=(225, 225, 1)):
  model.add(Conv2D(64, kernel_size, input_shape=input_shape, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(64, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(128, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(0.25))

  model.add(Conv2D(256, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(256, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(256, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(0.5))

  model.add(Conv2D(512, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(512, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(512, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
  model.add(Dropout(0.5))

  model.add(Conv2D(512, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(512, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(0.5))

  return model

def create_right_branch(model=Sequential(), kernel_size=(5, 5),  input_shape=(225, 225, 3)):
  model.add(Conv2D(24, kernel_size, input_shape=input_shape, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(5, 5), padding='same'))

  model.add(Conv2D(24, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(5, 5), padding='same'))

  model.add(Conv2D(24, kernel_size, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(AveragePooling2D(pool_size=(5, 5), padding='SAME'))

  return model

def create_model(model=Sequential(), input_shape=(225, 225, 3)):
  left_branch = create_left_branch(input_shape=input_shape)
  right_branch = create_right_branch(input_shape=input_shape)

  model.add(Merge([left_branch, right_branch], mode='concat'))
  model.add(Conv2D(1, (1, 1), padding='SAME'))
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

if __name__ == '__main__':
  im_generator = load_data()
  model = create_model()
  model.fit_generator(im_generator, epochs=100, verbose=True, steps_per_epoch=600)
  model.save('weights.h5')
