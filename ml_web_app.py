import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.title('Medical No-show Appointment Prediction Web App')
st.subheader('Dataset description')
st.markdown('''This dataset collects information from 100k medical appointments in Brazil 
and is focused on the question of whether or not patients show up for their appointment.
A number of characteristics about the patient are included in each row.
''')
st.markdown('This app will predict whether a patient will show up or not for his/her appointment')

df = pd.read_csv('KaggleV2-May-2016.csv')

# extracting only day, month and year values
df['ScheduledDay'] = df['ScheduledDay'].str[:10]
df['AppointmentDay'] = df['AppointmentDay'].str[:10]
# changing data type
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df["App_dayofweek"] = df["AppointmentDay"].dt.day_name()
# obtain waiting days by subtracting scheduled day from appointment day
df['waiting_days'] = df['AppointmentDay'] - df['ScheduledDay']
# convert the difference to number of days
df['waiting_days'] = df['waiting_days'].dt.days
df.drop(labels=df[df['waiting_days'] < 0].index, axis=0, inplace=True)
age_outlier = df.loc[df['Age'].isin([-1,115])] # assign all rows of dat with outliers to a new variable
df.drop(age_outlier.index, axis=0, inplace=True) # drop the outliers
# filling the values other than 0 and 1, with 1
df.loc[df['Handcap'].isin([2, 3, 4]), 'Handcap'] = 1
df.drop(['PatientId', 'AppointmentID', 'AppointmentDay', 'ScheduledDay','Neighbourhood'], axis=1, inplace=True)


st.subheader('Exploring the dataset')
st.write(f'**No of rows: {df.shape[0]}  \nNo of columns: {df.shape[1]}**')
st.sidebar.subheader('**Visual Exploration**')

col_vis = st.sidebar.selectbox(
    'Choose column you would like to visually explore',
    ('None','Age', 'Waiting days', 'Gender', 'Scholarship', 'SMS Received', 'Appointment Day'))

if col_vis == 'Age':
    sns.set_style('darkgrid')
    fig, axs = plt.subplots(1, 1, figsize =(10, 7), tight_layout = True)
    axs.hist(df['Age'])
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(fig)
elif col_vis == 'Waiting days':
    Show_WaitMean = df[df['No-show']=='No']['waiting_days'].mean()
    NoShow_WaitMean = df[df['No-show']=='Yes']['waiting_days'].mean()
    sns.set_style('darkgrid')
    plt.bar(['Show', 'No-show'], [Show_WaitMean, NoShow_WaitMean])
    plt.title('Mean Waiting Days')
    plt.ylabel('Mean')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
elif col_vis == 'Gender':
    sns.set_style('darkgrid')
    sns.countplot(x='Gender', hue='No-show', data=df)
    plt.title('No of Patients (Show/No-show) according to Gender')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
elif col_vis == 'SMS Received':
    sns.set_style('darkgrid')
    sns.countplot(x='SMS_received', hue='No-show', data=df)
    plt.title('No of Patients (Show/No-show) according to SMS Received')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
elif col_vis == 'Scholarship':
    sns.set_style('darkgrid')
    sns.countplot(x='Scholarship', hue='No-show', data=df)
    plt.title('No of Patients (Show/No-show) according to Scholarship')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
elif col_vis == 'Appointment Day':
    sns.set_style('darkgrid')
    sns.countplot(x='App_dayofweek', hue='No-show', data=df)
    plt.title('No of Patients (Show/No-show) according to Appointment Day')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Build Model
X = df.loc[:, df.columns != 'No-show']
Y = df['No-show']

X = pd.get_dummies(X)

# scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(X.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=X.columns)
X = df_scaled

# imputation
imputer = SimpleImputer(strategy='median').fit(X)
X = imputer.transform(X)


# train-test split
train_inputs, test_inputs, train_targets, test_targets = train_test_split(X, Y, test_size=0.2, random_state=101)

# model
st.markdown('#### Model Accuracy')
model = DecisionTreeClassifier(random_state=42)
model.fit(train_inputs, train_targets)
@st.cache
def build_model():
    train_preds = model.predict(train_inputs)
    acc = accuracy_score(train_targets, train_preds)
    return acc

clf = build_model()

st.write(f'**Model Accuracy: {clf*100:.2f}**')

def predict_input(new_input):
    input_df = pd.DataFrame([new_input])
    numeric_cols = input_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = input_df.select_dtypes('object').columns.tolist()

    scaler = MinMaxScaler().fit(input_df[numeric_cols].to_numpy())
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names(categorical_cols))

    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

prediction = st.subheader('Make a Prediction')
gender = st.selectbox('Gender', ('-', 'Male', 'Female'))
age = st.number_input('Age', 0)
sch = st.number_input('Scholarship (0-No, 1-Yes)', 0)
sms = st.number_input('SMS Received (0-No, 1-Yes)', 0)
hyp = st.number_input('Hypertension (0-No, 1-Yes)', 0)
alc = st.number_input('Alcoholic (0-No, 1-Yes)', 0)
dia = st.number_input('Diabetic (0-No, 1-Yes)', 0)
hand = st.number_input('Handicap (0-No, 1-Yes)', 0)
waiting_day = st.number_input('Waiting Days (No of days between scheduled and appointment)', 0)
app_day = st.selectbox('Appointment Day', ('-','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'))

    
if app_day != '-':
    new_input = {'Gender': gender, 'Age': age, 'Scholarship': sch, 'SMS_Received': sms, 
            'Hipertension': hyp, 'Alcoholism': alc, 'Diabetes': dia, 'Handcap': hand,
            'waiting_days': waiting_day, 'App_dayofweek': app_day}
    
    prediction = predict_input(new_input)
    st.markdown('**No-show Prediction: No- Will show up; Yes- will not show up**')
    st.write(f'**Prediction: {prediction[0]}  \nProbability: {prediction[1]:.2f}**')
