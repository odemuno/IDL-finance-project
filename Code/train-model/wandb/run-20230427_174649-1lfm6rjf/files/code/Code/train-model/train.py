# Import required libraries
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, TimeDistributed, Activation, Dot, Concatenate
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    df = df.dropna().reset_index(drop=True)
    
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


def load_and_train_model(X_train,y_train,X_test,y_test,model_type):
    model = None
    if model_type=='lstm':
        # Build the model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
    elif model_type=='lstm-attention':
        # Define the input layer
        inputs = Input(shape=(X_train.shape[1], 1))

        # Define the LSTM layer with return sequences set to True
        lstm_out = Bidirectional(LSTM(units=64, return_sequences=True))(inputs)

        # Define the attention mechanism
        attn_layer = TimeDistributed(Dense(1, activation='tanh'))(lstm_out)
        attn_weights = Activation('softmax', name='attention_weights')(attn_layer)
        context = Dot(axes=1, name='attention_dot')([attn_weights, lstm_out])
        context = Concatenate(axis=2)([context, lstm_out])

        # Define the remaining LSTM layers with return sequences set to False
        lstm_out2 = Bidirectional(LSTM(units=64, return_sequences=False))(context)
        lstm_out2 = Dropout(0.2)(lstm_out2)
        lstm_out3 = Bidirectional(LSTM(units=64, return_sequences=False))(lstm_out2)
        lstm_out3 = Dropout(0.2)(lstm_out3)

        # Define the output layer
        output = Dense(units=1)(lstm_out3)

        # Define the model with inputs and outputs
        model = Model(inputs=[inputs], outputs=[output])

    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')   

    # Define early stopping callback to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # train the model
    model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test),callbacks=[WandbCallback(monitor='val_loss', save_model=True),early_stopping])

    return model

def get_predictions(model,X_train,X_test,y_test,train_df,test_df,run_name):
    # Add predictions to final dataset
    # df = pd.concat([train_df,test_df],ignore_index=True)
    predicted_price_diff = model.predict(X_test)
    test_df['Predicted Price Diff'] = predicted_price_diff

    # Evaluate the model
    mse = mean_squared_error(y_test, predicted_price_diff)
    mae = mean_absolute_error(y_test, predicted_price_diff)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicted_price_diff)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')
    print(f'R-squared: {r2:.2f}')
    evaluation_df = pd.DataFrame({
        'Mean Squared Error':mse,
        'Mean Absolute Error':mae,
        'Root Mean Square Error':rmse,
        'R-squared':r2
    },index=[0])
    evaluation_df.to_csv('evaluation_metrics/metrics-'+run_name+'.csv')

    return test_df

def plot_graph(train_df,test_df,run_name):
    # Select rows where 'Date' starts with '2021'
    date_mask = train_df['Date'].str.startswith('2021')
    date_2021_df = train_df[date_mask]

    # Get the first row of the resulting DataFrame
    first_date_2021 = date_2021_df.iloc[0]['Date']

    # Plot the train_df data
    plt.figure(figsize=(10,6))
    plt.plot(train_df['Date'], train_df['Close'], color='blue', label='Actual PriceDiff 2020-2021',linewidth=1)

    # Plot the test_df data
    plt.plot(test_df['Date'], test_df['Close'], color='red', label='Actual PriceDiff 2022',linewidth=1)
    plt.plot(test_df['Date'], test_df['Predicted Price Diff'], color='green', label='Predicted PriceDiff 2022',linewidth=1)
    plt.xticks([train_df.iloc[0]['Date'],first_date_2021, test_df.iloc[0]['Date'],test_df.iloc[-1]['Date']])

    # Add title, x-label, and y-label
    plt.title('Predicted Stock Price Diff for '+run_name+' 2020-2022 (Train from 2020-2021 and Test 2022)')
    plt.xlabel('Date')
    plt.ylabel('Price Difference')

    # Add legend
    plt.legend()

    # Save plot
    plt.savefig('charts/'+run_name+'.png')


def main():
    # Load all arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis on Social Media Stock Data')
    parser.add_argument('wandb_api_key', type=str)
    parser.add_argument('wandb_run_name', type=str)
    parser.add_argument('train_dataset_path', type=str)
    parser.add_argument('test_dataset_path', type=str)
    parser.add_argument('model_type',type=str)
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
    model = load_and_train_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type=args.model_type)

    # Make predictions and evaluation and obtain the final dataframe
    test_df = get_predictions(model=model,X_train=X_train,X_test=X_test,y_test=y_test,train_df=train_df,test_df=test_df,run_name=args.wandb_run_name)

    # Plot the graph for predictions on 2022 stock price diff values
    plot_graph(train_df = train_df,test_df=test_df,run_name=args.wandb_run_name)

main()