#keras 练习1 ---- 使用LSTM对IMDB的分类进行预测(many to one 的RNN)

ChadCool原创 转载请注明出处
[https://chadCool.github.io/keras_example/lstm1](https://chadCool.github.io/keras_example/lstm1)

----------------------------

数据集来自 IMDB 的 25,000 条电影评论，以情绪（正面/负面）标记。每一条评论已经过预处理，并编码为词索引（整数）的[序列](https://keras.io/zh/preprocessing/sequence/)表示。为了方便起见，将词按数据集中出现的频率进行索引，例如整数 3 编码数据中第三个最频繁的词。这允许快速筛选操作，例如：「只考虑前 10,000 个最常用的词，但排除前 20 个最常见的词」。

```
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding, Masking
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
import pandas as pd

max_words = 5000

```
导入数据， 首次运行会下载， 比较慢
```
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=max_words,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
print("shape of x_train ", x_train.shape, " and shape of x_test is ", x_test.shape)
lens = [len(item) for item in x_train]
lens = pd.Series(lens)
print("小于500个字长度的评论占比", len(lens[lens < 800]) / len(lens))
```
输出如下：
shape of x_train  (25000,)  and shape of x_test is  (25000,)
小于500个字长度的评论占比 0.98 
```
#对数据进行截断, 
x_train = sequence.pad_sequences(x_train, 800)
x_test = sequence.pad_sequences(x_test, 800)
```
定义网络， 用最简单的方式， 参数随手写的， 先试试
```
model = Sequential()
model.add(Embedding(max_words, 64, input_length=800, mask_zero=True))
# model.add(Masking(0.0))
model.add(LSTM(30))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',  metrics=['accuracy'])
model.summary()
```
输出:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_4 (Embedding)      (None, 800, 64)           320000    
_________________________________________________________________
lstm_3 (LSTM)                (None, 30)                11400     
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 31        
=================================================================
Total params: 331,431
Trainable params: 331,431
Non-trainable params: 0
_________________________________________________________________
```
```
# 剩下就是漫长的训练时间
model.fit(x_train, y_train, epochs = 200, batch_size=128)
```
最终结果如下： 
```
...
...
...
#经过漫长的学习过程

Epoch 200/200
25000/25000 [==============================] - 211s 8ms/step - loss: 1.0970e-07 - acc: 1.0000
```
评估模型：
```
model.evaluate(x_test, y_test,batch_size=128, )
```
输出结果: 
```
25000/25000 [==============================] - 91s 4ms/step
[1.6355386052322387, 0.852760000038147]
```
看的出来， 输出结果准确率到了0.85, 相比训练集准确率到了1.0， 有过拟合的趋势
重新调整网络， 首先网络太慢了， 看看能否降低网络中LSTM层的数量, 增大训练集， 适当减小长句子的数量， 然后为网络层面加上dropout， 降低过拟合， 最后简单的测试之后， 数据如下
`loss: 0.1762 - acc: 0.9343 - val_loss: 0.3899 - val_acc: 0.8722`

最终网络如下：
```
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding, Masking
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
import pandas as pd

max_words = 5000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=max_words,
                                                      skip_top=0,
                                                      maxlen=200,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
                                                      
print("shape of x_train ", x_train.shape, " and shape of x_test is ", x_test.shape)
lens = [len(item) for item in x_train]
lens = pd.Series(lens)
print("小于500个字长度的评论占比", len(lens[lens < max_len]) / len(lens))

x_train = sequence.pad_sequences(x_train, max_len)
x_test = sequence.pad_sequences(x_test, max_len)

model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len, mask_zero=True))
# model.add(Masking(0.0))
model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',  metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs = 50, batch_size=32, validation_data=(x_test, y_test))

model.evaluate(x_test, y_test, batch_size=128)

```
训练了接近100次迭代后， 最终在验证集上的准确率没有明显提升， 我想要提升效果除了调整网络结构， 调整超参数之外， 更重要的一个方向估计是用现成的词向量来做优化吧。
这次尝试暂时到此为止。
> loss: 0.0057 - acc: 0.9986 - val_loss: 0.8961 - val_acc: 0.8763
