'''
LRN -> Batch Normalization
SGD -> Adam
'''

#import
import pandas as pd
import tensorflow as tf
import keras.backend as K

# overlapping minpool
def min_pool2d(x):
    max_val = K.max(x) + 1 # we gonna replace all zeros with that value
    # replace all 0s with very high numbers
    is_zero = max_val * K.cast(K.equal(x,0), dtype=K.floatx())
    x = is_zero + x
    # execute pooling with 0s being replaced by a high number
    min_x = -K.pool2d(-x, pool_size=(3, 3), strides=(2, 2))
    # depending on the value we either substract the zero replacement or not
    is_result_zero = max_val * K.cast(K.equal(min_x, max_val), dtype=K.floatx())
    min_x = min_x - is_result_zero
    return min_x # concatenate on channel

train_ds = tf.keras.utils.image_dataset_from_directory(
    './mask_trainset', image_size=(227, 227), seed=1337, batch_size=128, label_mode='binary', shuffle=True, validation_split=0.1, subset='training')

val_ds = tf.keras.utils.image_dataset_from_directory(
    './mask_trainset', image_size=(227, 227), seed=1337, batch_size=128, label_mode='binary', shuffle=True, validation_split=0.1, subset='validation')

test_ds = tf.keras.utils.image_dataset_from_directory('./mask_testset', image_size=(227, 227), batch_size=20, label_mode='binary')

# Input Layer
X = tf.keras.layers.Input(shape = [227, 227, 3])

# 1st Layer(Convolution Layer)
H = tf.keras.layers.Conv2D(96, kernel_size=11, strides=(4, 4), 
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))(X)

H = tf.keras.layers.BatchNormalization()(H)

H = tf.keras.layers.Activation('relu')(H)

H = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(H)
# H = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(H)
# H = min_pool2d(H)


# 2nd Layer(Convolution Layer)
H = tf.keras.layers.Conv2D(256, kernel_size=5, padding='same', 
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                           bias_initializer='ones')(H)

H = tf.keras.layers.BatchNormalization()(H)

H = tf.keras.layers.Activation('relu')(H)

H = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(H)
# H = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(H)
# H = min_pool2d(H)

# 3rd Layer(Convolution Layer)
H = tf.keras.layers.Conv2D(384, kernel_size=3, padding='same', activation='relu',
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))(H)

# 4th Layer(Convolution Layer)
H = tf.keras.layers.Conv2D(384, kernel_size=3, padding='same', activation='relu',
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                           bias_initializer='ones')(H)

# 5th Layer(Convolution Layer)
H = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu',
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                           bias_initializer='ones')(H)

H = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(H)
# H = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(H)
# H = min_pool2d(H)

# 주석 처리를 변경하여 실험.

H = tf.keras.layers.Flatten()(H)

# # 6th Layer(Fully connected Layer) 4096
H = tf.keras.layers.Dense(4096, kernel_initializer=
                          tf.random_normal_initializer(mean=0.0, stddev=0.01), 
                          bias_initializer='ones',
                          activation='relu')(H)

H = tf.keras.layers.Dropout(0.5)(H)

# 7th Layer(Fully connected Layer) 4096
H = tf.keras.layers.Dense(4096, kernel_initializer=
                          tf.random_normal_initializer(mean=0.0, stddev=0.01),
                          bias_initializer='ones',
                          activation='relu')(H)

H = tf.keras.layers.Dropout(0.5)(H)

# 8th Layer(Fully connected Layer) 
Y = tf.keras.layers.Dense(1, activation='sigmoid',
                          kernel_initializer=
                          tf.random_normal_initializer(mean=0.0, stddev=0.01))(H)

model = tf.keras.models.Model(X, Y)

model.compile(loss='binary_crossentropy', metrics = 'acc', optimizer = 'adam')

print("Fit: ") # batch 90
hist = model.fit(train_ds, validation_data=val_ds, epochs=90, batch_size=128)

print("Evalute: ")
score = model.evaluate(test_ds, batch_size=128)
print("정답률 = ", score[1], 'loss = ', score[0])

print("실제 사진 분류: ")
predict = model.predict(test_ds.take(1))
print(pd.DataFrame(predict).round(3))

print(model.summary())