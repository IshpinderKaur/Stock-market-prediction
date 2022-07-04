# import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, LSTM
from sklearn.preprocessing import StandardScaler

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

# Function that returns Preprocessed Data
def preprocess(df):
    #Close column is not required, so, dropping it
    df= np.delete(df,1,axis=2)
    # Also, date column is here acts as index and not required among 3*4 features ,so, dropping it
    df= np.delete(df,0,axis=2)
    # Converting all remaining feature values to float
    df=df.astype(float)
    # Standard scaler for 3-D data
    scale= StandardScaler()
    scale_df= scale.fit_transform(df.reshape(-1,df.shape[-1])).reshape(df.shape)
    return scale_df

# Function that returns the scaled data  for 2-D
def scale(df):
    scale= StandardScaler()
    df= scale.fit_transform(df)
    return df


if __name__ == "__main__":
	# 1. Load your saved model
    new_model = tf.keras.models.load_model('models/GROUP_7_RNN_model.h5')
    
	# 2. Load your testing data
    data = pd.read_csv('data/test_data_RNN.csv')
    test = data.copy()
    
    test = test.drop(test.iloc[:,0:1],axis=1)
    x_test=test.drop('y_test',axis=1)
    y_test=test[['y_test']]
    x_test=np.array(x_test)
    x_test=x_test.reshape(x_test.shape[0],3,6)
    
    # Getting preprocessed data for x_test
    x_test = preprocess(x_test)
    # Getting scaled data for y_test
    y_test = scale(y_test)
    
    ## training the network
    history_test=model.fit(x_test,y_test,epochs=100,batch_size=32,verbose=2)
    
    ##evaluating loss
    test_evaluation = model.evaluate(x_test,y_test,batch_size=32)
    print('The loss on test data is: ',test_evaluation)
    
	# 3. Run prediction on the test data and output required plot and loss
    ## Predictions
    pred_test= model.predict(x_test)
    
    ## Plotting the True and Predicted values
    plt.figure(figsize=(20,5))
    plt.plot(pred_test, label= 'Predicted')
    plt.plot(y_test,label='True')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Open price')
    plt.title('Predicted vs. Actual values of next day Open price of  Test data', fontsize=16, fontweight='bold')
    plt.show()