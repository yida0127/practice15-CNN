# practice15-CNN
use CNN to distinguish handwriting numbers from MNIST dataset

使用卷積神經網路(Convolutional Neural Network)辨別手寫數字(MNIST)
1. 初始準備 
   - 讀入tensorflow環境
        %env KERAS_BACKEND=tensorflow
2. 讀入MNIST資料庫
   - 輸入格式整理: 
        x_train = x_train.reshape(28,28,1)
        x_test = x_test.reshape(28,28,1)
   - 輸出格式整理: 載入np_utils將答案轉換成1-hot encoding格式
        y_train = np_utils.to_categorical(y_train,10)
        y_test = np_utils.to_categorical(y_test,10)
3. 打造神經網路
   - 決定神經網路架構並讀入相關套件
        from keras.datasets import Sequential
        from keras.layers import Dense, Activation, Flatten
        from keras.layers import Conv2D, MaxPooling2D
        from keras.optimizers import SGD
   - 建構神經網路
        要添加三層layers，最後要拉平最後組裝
4. 檢視成果
   - model.summary()
5. 訓練神經網路
   - 將x_train, y_train丟入模型中訓練 
        model.fit(x_train,y_train,batch_size=100,epochs=12)
   - batch_size 每次訓練的資料量
   - epochs 訓練次數
6. 試用成果
   - score = model.evaluate(x_test)
   - score[0]:loss ; score[1]:準確度
7. 將訓練好的神經網路分別存下來
   - 存本體 
        model_json = model.to_json()
        open('XXXX.json','w').write(model.json)
   - 存權重 
        model.save_weights('XXXX.h5')
        
*****
如果訓練後的準確度不好，可以更改以下
1. CNN層數
2. 更改每層filter個數, kernal_size
3. Activation Function可改為'relu', 'selu', 'learkyrelu'
4. Optimizer可改為'RMSProp','adam'
5. 更改learning rate
6. 你想改的任何東西，只要能提升準確度都好
*****
