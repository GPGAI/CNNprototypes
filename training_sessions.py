from tensorflow import keras
from keras.callbacks import EarlyStopping

def train_model_1(model):
    opt = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=opt,
        loss={
            'keys_output': 'hinge',
            'click_output': 'categorical_crossentropy',
            'xcor_output': 'mse',
            'ycor_output': 'mse'
        },
        metrics={
            'keys_output': ['accuracy'],
            'click_output': ['accuracy'],
            'xcor_output': ['mae'],
            'ycor_output': ['mae']
        }
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        x=train_x,
        y=train_y,
        validation_data=(val_x, val_y),
        epochs=100,
        callbacks=[early_stopping]
    )

    return model


def train_model_2(model):
    opt = keras.optimizers.Adam(learning_rate=0.1)

    model.compile(
        optimizer=opt,
        loss={
            'keys_output': 'binary_crossentropy',
            'click_output': 'categorical_crossentropy',
            'xcor_output': 'mae',
            'ycor_output': 'mae'
        },
        metrics={
            'keys_output': ['accuracy'],
            'click_output': ['accuracy'],
            'xcor_output': ['mae'],
            'ycor_output': ['mae']
        }
    )

    return model