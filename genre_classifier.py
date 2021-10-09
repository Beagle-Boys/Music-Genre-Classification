import json
import numpy as np
from sklearn.model_selection import train_test_split
# 50
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "DATA.json"

def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
        
    # convert lists to np arrays
    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])
    
    return inputs, targets

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
    
    fig.savefig('./plots/ann_w_overfitting.png')
    
    plt.show()


if __name__ == '__main__':
    # load data
    inputs,targets = load_data(DATASET_PATH)
    
    # split data in train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, 
                                                                              targets, 
                                                                              test_size=0.3)
    
    # build network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        
        # 1st hidden layer
        keras.layers.Dense(512, activation='relu'),
        
        # 2nd hidden layer
        keras.layers.Dense(256, activation='relu'),
        
        # 3rd hidden layer
        keras.layers.Dense(64, activation='relu'),
        
        # Output layer
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.summary()
    
    keras.utils.plot_model(model, to_file='./plots/simple_ann.png', show_shapes=False)
    
    # train network
    history = model.fit(inputs_train, targets_train, 
              validation_data=(inputs_test, targets_test), 
              epochs=50,
              batch_size=32)
    
    plot_history(history)