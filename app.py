import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Chance in the Titanic Journey")

pclass = st.slider('Enter the passenger class for the user', 1,3)
sex = st.selectbox('Enter the passenger Gender', ['male', 'female'])
sibsp = st.slider('Enter the Passenger total number of Sibling and Spouse', 1,8)
parch = st.slider('Enter the Passenger total number of Parent and Children', 0,6)
fare = st.number_input("Enter the Fare of the Pssenger")
embarked = st.selectbox('Enter the Passenger Station from where they started the journey', ['Chebourg', 'Queenstoen', 'Southampton'])

data = pd.DataFrame([{'Pclass': pclass, 'Sex': sex, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}])

model = load_model('model.h5')

with open('label_encoder.pkl', 'rb') as file:
    label = pickle.load(file)


with open('onehot_encoder.pkl', 'rb') as file:
    onehot = pickle.load(file)


with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


data['Sex'] = label.transform(data['Sex'])

embark = onehot.transform(data[['Embarked']])
embark= pd.DataFrame(embark, columns=onehot.get_feature_names_out())

data = pd.concat([data.drop(columns=['Embarked']), embark], axis=1)

data[['Pclass', 'SibSp', 'Parch', 'Fare']] = scaler.transform(data[['Pclass', 'SibSp', 'Parch', 'Fare']])

y = model.predict(data)

y = y[0][0]

def chance(y):
    if y>0.5:
        return 'Passenger will survived.'
    else:
        return "Passenger won't survived."

if st.button('Predict Survival Chance'):
    st.write("Probability of Passanger Survival Chance", y)
    st.write(chance(y))

