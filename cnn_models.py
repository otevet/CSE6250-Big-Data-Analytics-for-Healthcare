from keras.models import *
from keras.layers import *

def cnn_text(input_shape, output_shape, embedding_layer):
    sequence_input = Input(shape=input_shape, dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', padding='same')(embedded_sequences)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(35, padding='same')(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(output_shape, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])

    return model