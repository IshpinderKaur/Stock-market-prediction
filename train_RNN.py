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

#Function that returns the feature and target arrays for RNN model
def create_data(df,past_days,future_day):
    x, y =[],[]
    for i in range(past_days, len(df)-future_day+1):
        x.append(df.iloc[i-past_days:i,0:df.shape[1]])
        y.append(df.iloc[i+future_day-1:i+future_day,3])
    return np.array(x),np.array(y)

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

# loading the data
data= pd.read_csv('data/q2_dataset.csv')
df= data.copy()
    
# Future day whose price is to be predicted
future_day=1
# Count of past days on basis of which target is to be predicted
past_days=3
    
# Data creation
x_data , y_data = create_data(df,past_days,future_day)
    
# Randomization and Train_test_split of data
x_train , x_test , y_train ,y_test =train_test_split(x_data,y_data,test_size=0.3, random_state=42)

# Saving csv file
train = pd.DataFrame( x_train.reshape(x_train.shape[0],18))
train[['y_train']]= y_train
train.to_csv('data/train_data_RNN.csv')
test = pd.DataFrame( x_test.reshape(x_test.shape[0],18))
test[['y_test']]= y_test
test.to_csv('data/test_data_RNN.csv')

if __name__ == "__main__": 
	# 1. load your training data
    data = pd.read_csv('data/train_data_RNN.csv')
    train= data.copy()
    
    train = train.drop(train.iloc[:,0:1],axis=1)
    x_train=train.drop('y_train',axis=1)
    y_train=train[['y_train']]
    x_train=np.array(x_train)
    x_train=x_train.reshape(x_train.shape[0],3,6)
    
    # Getting preprocessed data for x_train
    x_train = preprocess(x_train)
    # Getting scaled data for y_train
    y_train = scale(y_train)
    
	# 2. Train your network
    # Model building
    model=Sequential()
    model.add(LSTM(64,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2]),return_sequences=True))
    model.add(LSTM(32,activation='relu',return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1]))
    model.compile(loss='mse',optimizer='adam')
    model.summary()
	# 		Make sure to print your training loss within training to show progress
         # Training the network and printing respective training loss
    history=model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=2)
    

	# 		Make sure you print the final training loss
        # printing final training loss
    print('The final training loss is: ',history.history['loss'][-1])

	# 3. Save your model
    model.save('models/GROUP_7_RNN_model.h5')
