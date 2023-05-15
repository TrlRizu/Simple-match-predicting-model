#Importing the libs

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout



# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    #Drop unwanted data
    df = data.drop(columns=['possession team1','possession team2','possession in contest','date','hour','category','total attempts team1','total attempts team2','conceded team1','conceded team2','goal inside the penalty area team1','goal inside the penalty area team2','goal outside the penalty area team1','goal outside the penalty area team2','assists team1','assists team2','on target attempts team1','on target attempts team2','off target attempts team1','off target attempts team2','attempts inside the penalty area team1','attempts inside the penalty area  team2','attempts outside the penalty area  team1','attempts outside the penalty area  team2','left channel team1','left channel team2','left inside channel team1','left inside channel team2','central channel team1','central channel team2','right inside channel team1','right inside channel team2','right channel team1','right channel team2','total offers to receive team1','total offers to receive team2','inbehind offers to receive team1','inbehind offers to receive team2','inbetween offers to receive team1','inbetween offers to receive team2','infront offers to receive team1','infront offers to receive team2','receptions between midfield and defensive lines team1','receptions between midfield and defensive lines team2','attempted line breaks team1','attempted line breaks team2','completed line breaksteam1','completed line breaks team2','attempted defensive line breaks team1','attempted defensive line breaks team2','completed defensive line breaksteam1','completed defensive line breaks team2','yellow cards team1','yellow cards team2','red cards team1','red cards team2','fouls against team1','fouls against team2','offsides team1','offsides team2','passes team1','passes team2','passes completed team1','passes completed team2','crosses team1','crosses team2','crosses completed team1','crosses completed team2','switches of play completed team1','switches of play completed team2','corners team1','corners team2','free kicks team1','free kicks team2','penalties scored team1','penalties scored team2','goal preventions team1','goal preventions team2','own goals team1','own goals team2','forced turnovers team1','forced turnovers team2','defensive pressures applied team1','defensive pressures applied team2'])

    # add a new column to represent the target variable
    #1 if the home team won, -1 if the away team won, and 0 if the match was a draw.
    #Creating the target variable
    df['winner'] = df.apply(lambda row: 1 if row['number of goals team1'] > row['number of goals team2'] else (-1 if row['number of goals team1'] < row['number of goals team2'] else 0), axis=1)


    # One-hot encode categorical features
    X = pd.get_dummies(df[['team1', 'team2']])
    Y = df['winner']

    return X,Y

def split_data(X, Y, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    return X_train, X_test, Y_train, Y_test

def create_model(input_dim):
    #Create the neural network model
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_model(model, X_train, Y_train, epochs=25, batch_size=16, verbose=1):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     
     # Train the model
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Loss:", loss)
    print("Accuracy:", accuracy)


def main():
    file_path = "Fifa_world_cup_matches.csv"

    # Load the data
    data = load_data(file_path)

    # Preprocess the data
    X, Y = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Create the model
    model = create_model(X.shape[1])

    # Train the model
    train_model(model, X_train, Y_train)

    # Evaluate the model
    evaluate_model(model, X_test, Y_test)

if __name__ == "__main__":
    main()