from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import config

# model
FILTERS = config.FILTERS
INPUT_DIM = config.INPUT_DIM
ACTIVATION = config.ACTIVATION
INITIALIZER = config.INITIALIZER
LR_INIT = config.LR_INIT

def build_model():
    model = Sequential()
    
    model.add(Dense(FILTERS, input_dim = INPUT_DIM ,activation = ACTIVATION, kernel_initializer = INITIALIZER ))
    model.add(Dense(1, activation = 'linear' ))

    model.compile(loss= 'mean_squared_error', optimizer=Adam(lr= LR_INIT), metrics= ['mean_squared_error'])

    return model

model = build_model()