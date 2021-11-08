# Methodology

## Analyse

* **Waveform**
    
    ![Waveform](./plots/waveform.png)
* **FFT Spectrum** 
    
    ![FFT](./plots/fft.png)
* **Log Spectogram** 
    
    ![Log Spectogram](./plots/log_spectogram.png)
* **MFCC** 
    
    ![MFCC](./plots/mfcc.png)

***

## Preprocess

* divided each `.wav` file into 5 segments to increse training data.
* exctracted mfcc for each segment.
* stored the complete dataset in format :
    ```json
    {
        "mapping" : ["classical","blues",...],
        "mfcc": [[[...],[...],...,[...]],...],
        "labels": [0,2,...]
    }
    ```
***

### Simple ANN

* **Architecture** 
    
    ![Archiecture](./plots/simple_ann.png)

* **Result** 
    
    ![Plots](./plots/ann_w_overfitting.png)

* **Comments** : 
    - As we can see, the model is overfitting.

***

### Simple ANN with Dropout and Kernel Regularization

* **Architecture** 
    
    ![Architecture](./plots/simple_ann_solved_overfitting.png)

* **Result** 
    
    ![Plots](./plots/ann_w_overfitting_solved.png)

* **Comments** :
    - The model is not overfitting anymore.

***

### Convolutional Neural Network

* **Architecture** 
    
    ![Architecture](./plots/cnn.png)

* **Result** 
    
    ![Plots](./plots/cnn_history.png)

* **Comments** :
    - CNN performs very well.
    - The training data had to be reshaped since CNN required 3 dimensional input.
    - Fastest

***

### Recurrent Neural Network (LSTM)

* **Architecture** 
    
    ![Architecture](./plots/lstm.png)

* **Result** 
    
    ![Plots](./plots/lstm_history.png)

* **Comments** :
    - LSTM performs very well.
    - Slowest

***

### Recurrent Neural Network (GRU)

* **Architecture** 
    
    ![Architecture](./plots/gru.png)

* **Result** 
    
    ![Plots](./plots/gru_history.png)

* **Comments** :
    - GRU performs very well. Better than LSTM.
    - Slowest (as slow as LSTM)