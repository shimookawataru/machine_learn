import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Sequence

## start create X, y
X_0 = np.random.normal(
    0.1,
    0.1,
    200,
)
X_1 = np.random.normal(
    0.2,
    0.1,
    200,
)
X = np.concatenate([X_0, X_1])
X = X.reshape((400, 1))
y = np.zeros(400)
y[200:] = 1
y = to_categorical(y, 2)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
## end create X, y

## start create model
def create_model():
    model = Sequential()
    model.add(
        Dense(
            2,
            activation='softmax',
        )
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
## end create model


## start create gen
class gen(Sequence):

    def __init__(self, x_set, y_set, batch_size=10):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
## end create gen

## start fit
g = gen(X_train, y_train)
m = create_model()
m.fit_generator(
    g,
    epochs = 100
)
# p = m.predict_proba(X_test)
predict_prob = m.predict(X_test)
p = np.argmax(predict_prob,axis=1)

score = roc_auc_score(
    y_test[:,1],
    p,
)
print(score)
## end fit