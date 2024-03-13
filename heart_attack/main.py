# %%
import pandas as pd
# %%
df = pd.read_csv("Heart Attack Data Set.csv")
# %%
df
# %%
x = df[["age","sex","chol","cp"]]
y = df["target"]
# %%
x
# %%
from sklearn.model_selection import train_test_split
# %%
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=10)
# %%
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError
# %%
model = Sequential()
# %%
# Input Layers
model.add(Input(shape=X_train.shape[1]))

# Hidden Layers
model.add(Dense(150, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(100, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(70, activation="relu"))

model.add(Dense(50, activation="relu"))

model.add(Dense(30, activation="relu", use_bias=False))

# Output Layers
model.add(Dense(1))
# %%
model.compile(optimizer=Adam(), loss=MeanAbsoluteError(), metrics=[MeanSquaredError()])
# %%
stop = EarlyStopping(monitor="val_loss", verbose=1, patience=25, min_delta=3)
# %%
model.fit(X_train, y_train, callbacks=stop, verbose=1, epochs=250, validation_split=0.2, batch_size=32)