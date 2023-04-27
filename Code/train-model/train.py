# Import required libraries
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
import argparse

# Wandb setup
def wandb_setup(api_key,run_name):
    wandb.login(key=api_key)
    run = wandb.init(
            name = run_name, ## Wandb creates random run names if you skip this field
            project = "idl-finance" ### Project should be created in your wandb account 
    )    

'''
 Load the train and test datasets
 Train dataset : 2020-2021
 Test dataset : 2022
'''
def load_datasets(train_dataset_path,test_dataset_path):
    # Load train dataset
    train_df = pd.read_csv(train_dataset_path)
    # Load test dataset
    test_df = pd.read_csv(test_dataset_path)
    return train_df,test_df

# Create 7 new features for the dataset
def feature_engineering(df):
    # Create lagged features for the price_diff values
    for i in range(1, 8):
        df[f'PriceDiff_{i}'] = df['Close'].shift(i)

    # Remove missing values
    df.dropna(inplace=True)
    return df

'''
 Spit the train and test dataframes into X and y dataframes
 X : the feature matrix
 y : the labels (stock price difference)
'''
def train_test_split(train_df,test_df):
    X_train = train_df[['sentiment_score', 'PriceDiff_1', 'PriceDiff_2', 'PriceDiff_3', 'PriceDiff_4', 'PriceDiff_5', 'PriceDiff_6', 'PriceDiff_7']]
    y_train = train_df.Close
    X_test = test_df[['sentiment_score', 'PriceDiff_1', 'PriceDiff_2', 'PriceDiff_3', 'PriceDiff_4', 'PriceDiff_5', 'PriceDiff_6', 'PriceDiff_7']]
    y_test = test_df.Close
    return X_train,y_train,X_test,y_test

def load_and_train_model(X_train,y_train,X_test,y_test):
    # Build the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')   

    # train the model
    model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(X_test, y_test),callbacks=[WandbCallback(monitor='val_loss', save_model=True)])

    return model

def get_predictions(model,X_train,X_test,train_df,test_df):
    # Get predictions for train and test dataset
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    predicted_price_diff = np.concatenate((predictions_train,predictions_test),axis=0)

    # Add predictions to final dataset
    df = pd.concat([train_df,test_df])
    df['Predicted Price Diff'] = predicted_price_diff

    return df

def plot_graph(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(y='PriceDiff',color='green',ax=ax,label='Actual Price Diff')
    df.loc['2022-01-01':, 'Predicted Price Diff'].plot(ax=ax, color='red', label='Predicted Price Diff for 2022')
    ax.set_title('Predicted Stock Price Diff for 2020-2022 (Train from 2020-2021 and Test 2022)')


def main():
    # Load all arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis on Social Media Stock Data')
    parser.add_argument('wandb_api_key', type=str)
    parser.add_argument('wandb_run_name', type=str)
    parser.add_argument('train_dataset_path', type=str)
    parser.add_argument('test_dataset_path', type=str)
    args = parser.parse_args()

    # Perform the wandb setup
    wandb_setup(api_key=args.wandb_api_key,run_name=args.wandb_run_name)

    # Load the datasets
    train_df, test_df = load_datasets(train_dataset_path=args.train_dataset_path,test_dataset_path=args.test_dataset_path)

    # Perform feature engineering for train and test datasets
    train_df = feature_engineering(df=train_df)
    test_df = feature_engineering(df=test_df)

    # Perform features and labels split for train and test datasets
    X_train, y_train, X_test, y_test = train_test_split(train_df=train_df,test_df=test_df)

    # Load the LSTM model and train it
    model = load_and_train_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)

    # Make predictions and obtain the final dataframe
    final_df = get_predictions(model=model,X_train=X_train,X_test=X_test,train_df=train_df,test_df=test_df)

    # Plot the graph for predictions on 2022 stock price diff values
    plot_graph(final_df)

main()