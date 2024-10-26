# bug: NotImplementedError: Cannot convert a symbolic Tensor (simple_rnn/strided_slice:0) to a numpy array.
# fix: pip install numpy=1.19

# 導入函式庫
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import GRU, Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
(X_train, y_train), (X_test, y_test_org) = mnist.load_data()

# 將 training 的 input 資料轉為3維，並 normalize 把顏色控制在 0 ~ 1 之間
X_train = X_train.reshape(-1, 28, 28) / 255.0
X_test = X_test.reshape(-1, 28, 28) / 255.0


# 建立簡單的線性執行的模型
model = Sequential()
# 加 RNN 隱藏層(hidden layer)
# 必須是 3 dimension
model.add(GRU(units=256, input_shape=(28, 28)))

# 加 output 層
model.add(Dense(units=10, activation="softmax"))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
LR = 0.001  # Learning Rate
adam = Adam(LR)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000001000，即第7個值為 1
y_TrainOneHot = to_categorical(y_train)
y_TestOneHot = to_categorical(y_test_org)

# 將 training 的 input 資料轉為2維
X_train_2D = X_train.reshape(60000, 28, 28)
X_test_2D = X_test.reshape(10000, 28, 28)

x_Train_norm = X_train_2D / 255
x_Test_norm = X_test_2D / 255

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(
    x=x_Train_norm,
    y=y_TrainOneHot,
    validation_split=0.2,
    epochs=10,
    batch_size=800,
    verbose=2,
)

# 顯示訓練成果(分數)
loss, accuracy = model.evaluate(x_Test_norm, y_TestOneHot)
print(f"test loss: {loss}  test accuracy: {accuracy}")

# 預測(prediction)
X = x_Test_norm[0:20, :]
predictions = np.argmax(model.predict(x_Test_norm[0:20]), axis=-1)
# get prediction result
print("actual :", y_test_org[0:20])
print("predict:", predictions)
