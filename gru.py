import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.keras as keras

DATASET_PATH = "DATA.json"

def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
        
    # convert lists to np arrays
    X = np.array(data['mfcc'])
    Y = np.array(data['labels'])
    
    return X, Y

def plot_history(history):
    
    fig, axs = plt.subplots(2)
    
    # create accuracy sublpot
    axs[0].plot(history.history['accuracy'], label="train accuracy")
    axs[0].plot(history.history['val_accuracy'], label="test accuracy")
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc="lower right")
    axs[0].set_title('Accuracy evaluation')
    
    # create loss sublpot
    axs[1].plot(history.history['loss'], label="train loss")
    axs[1].plot(history.history['val_loss'], label="test loss")
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc="upper right")
    axs[1].set_title('Loss evaluation')
    
    fig.savefig("./plots/gru_history.png")
    plt.show()

def prepare_datasets(test_size,validation_size):
    # load data
    X, y = load_data(DATASET_PATH)
    
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    
    # create a model
    model = keras.Sequential()
    
    # 2 LSTM layers
    model.add(keras.layers.GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.GRU(64))
    
    # dense layer
    model.add(keras.layers.Dense(64, activation='relu')),#, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))
    
    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model

def predict(model,X,y):
    
    X = X[np.newaxis,...]
    
    # prediction = [[0,1, 0.2, ...]]
    prediction = model.predict(X) # X -> (1, 130 ,13 ,1)
    
    predicted_index = np.argmax(prediction,axis=1) # [3]
    
    print("Predicted Index : {}, Actual Index : {}".format(predicted_index,y))

if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    
    # build the LSTM network
    input_shape = (X_train.shape[1], X_train.shape[2]) # (130, 13)
    model = build_model(input_shape)
    
    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    keras.utils.plot_model(model, to_file='./plots/gru.png', show_shapes=True)
    
    # train LSTM
    history = model.fit(X_train,y_train,validation_data=(X_validation,y_validation), 
                        batch_size=32,
                        epochs=50)
    
    # plot history
    plot_history(history)
    
    # evaluate CNN on test set
    test_err, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print('Test accuracy:', test_acc)
    print('Test error:', test_err)
    
    # make predictions on sample
    X = X_test[100]
    y = y_test[100]
    
    predict(model, X, y)
    
    model.save('gru_model')