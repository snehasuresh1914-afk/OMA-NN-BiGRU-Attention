# =========================================================
# OMA-NN-BiGRU-Attention (FINAL PAPER-CONSISTENT VERSION)
# =========================================================

import numpy as np
import pandas as pd
import random as r
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr
from scipy.special import expit
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, GRU, Bidirectional, concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# =========================================================
# DATA LOADING (FIXED SPLIT 60-20-20)
# =========================================================
def load_data(path):
    df = pd.read_csv(path)

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))

    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    def split(dataset):
        xi = dataset.iloc[:, 0:8]
        xg = dataset.iloc[:, 8:103]
        xd = xg.values.reshape(xg.shape[0], 3, int(xg.shape[1]/3))
        y = dataset.iloc[:, 104:]
        return xd, xi, y

    train_xd, train_xi, train_y = split(train)
    val_xd, val_xi, val_y = split(val)
    test_xd, test_xi, test_y = split(test)

    return train_xd, val_xd, test_xd, train_xi, val_xi, test_xi, train_y, val_y, test_y

# =========================================================
# ATTENTION
# =========================================================
class AttentionLayer(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units))
        self.b = self.add_weight(shape=(self.units,))
        self.u = self.add_weight(shape=(self.units, 1))

    def call(self, x):
        score = K.tanh(K.dot(x, self.W) + self.b)
        weights = K.softmax(K.dot(score, self.u), axis=1)
        context = weights * x
        return K.sum(context, axis=1)

# =========================================================
# MODEL
# =========================================================
def build_model(params, train_xd, train_xi):

    seq_input = Input(shape=(train_xd.shape[1], train_xd.shape[2]))
    x = Bidirectional(GRU(int(params[1]), return_sequences=True))(seq_input)
    x = Dropout(params[5])(x)
    x = Bidirectional(GRU(int(params[1]), return_sequences=True))(x)
    x = Dropout(params[5])(x)

    x = AttentionLayer(int(params[7]))(x)
    x = Dense(int(params[3]), activation='linear')(x)

    static_input = Input(shape=(train_xi.shape[1],))
    y = Dense(int(params[0]), activation='relu')(static_input)
    y = Dropout(params[4])(y)
    y = Dense(int(params[0]), activation='relu')(y)
    y = Dropout(params[4])(y)
    y = Dense(int(params[2]), activation='linear')(y)

    merged = concatenate([x, y], name="Concatenate")
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(params[6]), loss='mse')

    return model

# =========================================================
# OBJECTIVE FUNCTION (USES VALIDATION SET)
# =========================================================
def objective_function(x, data):
    train_xd, val_xd, train_xi, val_xi, train_y, val_y = data

    model = build_model(x, train_xd, train_xi)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    history = model.fit(
        [train_xd, train_xi], train_y,
        epochs=1000,
        batch_size=train_y.shape[0],
        validation_data=([val_xd, val_xi], val_y),
        shuffle=False,
        verbose=0,
        callbacks=[callback]
    )

    K.clear_session()
    return np.min(history.history['val_loss'])

# =========================================================
# OMA 1
# =========================================================
def run_oma1(data):

    train_xd, val_xd, test_xd, train_xi, val_xi, test_xi, train_y, val_y, test_y = data

    oma_data = (train_xd, val_xd, train_xi, val_xi, train_y, val_y)

    nVar, npop, maxiter = 8, 10, 50
    Ub = np.array([200,100,20,20,0.5,0.5,0.1,64])
    Lb = np.array([5,5,1,1,0,0,0.0001,4])

    x = np.zeros((npop, nVar))
    x[0] = np.random.rand(nVar)
    for i in range(1, npop):
        x[i] = x[i-1]*(1-x[i-1])

    x = Lb + x*(Ub-Lb)

    fit = np.array([objective_function(ind, oma_data) for ind in x])

    for it in range(maxiter):
        best_idx = np.argmin(fit)
        bestsol = x[best_idx]

        for i in range(npop):

            xnew = bestsol + np.random.rand(nVar)*1.4*x[i]
            xnew = np.clip(xnew, Lb, Ub)

            fitnew = objective_function(xnew, oma_data)
            if fitnew < fit[i]:
                x[i], fit[i] = xnew, fitnew

            j = r.randint(0, npop-1)
            while j == i:
                j = r.randint(0, npop-1)

            space = x[j]-x[i] if fit[i]>=fit[j] else x[i]-x[j]

            xnew = x[i] + np.random.rand(nVar)*0.55*space
            xnew = np.clip(xnew, Lb, Ub)

            fitnew = objective_function(xnew, oma_data)
            if fitnew < fit[i]:
                x[i], fit[i] = xnew, fitnew

        print(f"OMA1 Iter {it+1} Best: {np.min(fit)}")

    return x[np.argmin(fit)]

# =========================================================
# OMA 2
# =========================================================
def run_oma2(concat_output, targets):

    def fitness(w):
        return np.mean((targets - expit(np.dot(concat_output, w)))**2)

    nVar = concat_output.shape[1]
    npop, maxiter = 100, 100

    x = np.random.rand(npop, nVar)
    fit = np.array([fitness(ind) for ind in x])

    for it in range(maxiter):
        best_idx = np.argmin(fit)
        bestsol = x[best_idx]

        for i in range(npop):

            xnew = bestsol + np.random.rand(nVar)*1.4*x[i]
            xnew = np.clip(xnew, -1, 1)

            fitnew = fitness(xnew)
            if fitnew < fit[i]:
                x[i], fit[i] = xnew, fitnew

            j = r.randint(0, npop-1)
            while j == i:
                j = r.randint(0, npop-1)

            space = x[j]-x[i] if fit[i]>=fit[j] else x[i]-x[j]

            xnew = x[i] + np.random.rand(nVar)*0.55*space
            xnew = np.clip(xnew, -1, 1)

            fitnew = fitness(xnew)
            if fitnew < fit[i]:
                x[i], fit[i] = xnew, fitnew

        print(f"OMA2 Iter {it+1} Best: {np.min(fit)}")

    return x[np.argmin(fit)]

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    import os

    # Check dataset exists
    if not os.path.exists("ESTC Denorm.csv"):
        print("Dataset not found. Please provide ESTC Denorm.csv")
        exit()

    # Now safe to load
    data = load_data("ESTC Denorm.csv")

    print("Running OMA 1...")
    best_params = run_oma1(data)

    train_xd, val_xd, test_xd, train_xi, val_xi, test_xi, train_y, val_y, test_y = data

    model = build_model(best_params, train_xd, train_xi)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_weights.keras',
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss'
    )

    model.fit(
        [train_xd, train_xi], train_y,
        epochs=500,
        batch_size=train_y.shape[0],
        validation_data=([val_xd, val_xi], val_y),
        shuffle=False,
        callbacks=[checkpoint]
    )

    model.load_weights('best_weights.keras')

    concat_model = Model(inputs=model.input,
                         outputs=model.get_layer("Concatenate").output)

    train_concat = concat_model.predict([train_xd, train_xi])
    test_concat = concat_model.predict([test_xd, test_xi])

    best_weights = run_oma2(train_concat, train_y.values.reshape(-1))

    y_pred = expit(np.dot(test_concat, best_weights))

    # Denormalization
    max_y = test_y.max().values[0]
    min_y = test_y.min().values[0]
    test_y_act = test_y*(max_y-min_y)+min_y
    y_pred_act = y_pred*(max_y-min_y)+min_y

    test_y_act = test_y_act.values.reshape(-1)

    R = pearsonr(test_y_act, y_pred_act)[0]
    R2 = r2_score(test_y_act, y_pred_act)
    MSE = mean_squared_error(test_y_act, y_pred_act)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(test_y_act, y_pred_act)
    MAPE = mean_absolute_percentage_error(test_y_act, y_pred_act)
    
    print("\nFinal Results:")
    print("R:", R)
    print("R2:", R2)
    print("MSE:", MSE)
    print("RMSE:", RMSE)
    print("MAE:", MAE)
    print("MAPE:", MAPE)
    
    ri = (R + R2 + (1 - RMSE) + (1 - MAE) + (1 - MAPE)) / 5
    print("RI:", ri)