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
from sklearn.metrics import confusion_matrix
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

df.drop(['PatientId', 'AppointmentID', 'AppointmentDay', 'ScheduledDay','Neighbourhood'], axis=1, inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes('object').columns.tolist()

# scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[numeric_cols].to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)
df[numeric_cols] = df_scaled

# imputation
imputer = SimpleImputer(strategy='mean').fit(df[numeric_cols])
df[numeric_cols] = imputer.transform(df[numeric_cols])

# train-test split
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

input_cols = ['Gender','Age', 'Scholarship', 'Hipertension', 'Diabetes',
       'Alcoholism', 'Handcap', 'SMS_received', 'waiting_days', 'App_dayofweek']
target_col = 'No-show'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# encoding categorical variable
cat_cols = train_inputs.select_dtypes('object').columns.tolist()
num_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(train_inputs[cat_cols])
encoded_cols = list(encoder.get_feature_names(cat_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[cat_cols])
train_inputs = train_inputs[num_cols + encoded_cols]

cat_cols = val_inputs.select_dtypes('object').columns.tolist()
num_cols = val_inputs.select_dtypes(include=np.number).columns.tolist()
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(val_inputs[cat_cols])
encoded_cols = list(encoder.get_feature_names(cat_cols))
val_inputs[encoded_cols] = encoder.transform(val_inputs[cat_cols])
val_inputs = val_inputs[num_cols + encoded_cols]

cat_cols = test_inputs.select_dtypes('object').columns.tolist()
num_cols = val_inputs.select_dtypes(include=np.number).columns.tolist()
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(test_inputs[cat_cols])
encoded_cols = list(encoder.get_feature_names(cat_cols))
test_inputs[encoded_cols] = encoder.transform(test_inputs[cat_cols])
test_inputs = test_inputs[num_cols + encoded_cols]
# model
select_model = st.sidebar.selectbox('Choose a Classifier model', ('None', 'Random Forest', 'Decision Tree'))

@st.cache
def model_acc(inputs, targets, name=''):
    model.fit(inputs, targets)
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets, preds)
    return accuracy

@st.cache
def get_params(s_model):
    params = dict()
    if s_model == 'Random Forest':
        max_dept = st.sidebar.slider('max_dept', 1, 50)
        max_leaf_nodes = st.sidebar.slider('max_leaf_nodes', 2**1, 2**15)
        n_estimators = st.sidebar.slider('n_estimators', 5, 500)
        params['max_dept'] = max_dept
        params['max_leaf_nodes'] = max_leaf_nodes
        params['n_estimators'] = n_estimators
    elif s_model == 'Decision Tree':
        max_dept = st.sidebar.slider('max_dept', 1, 50)
        max_leaf_nodes = st.sidebar.slider('max_leaf_nodes', 2**1, 2**15)
        params['max_dept'] = max_dept
        params['max_leaf_nodes'] = max_leaf_nodes 
    return params 

params = get_params(select_model)

if params:
    if select_model == 'Random Forest':
        model = RandomForestClassifier(max_depth= params['max_dept'], max_leaf_nodes=params['max_leaf_nodes'], n_estimators= params['n_estimators'], random_state=1234)
        acc = model_acc(train_inputs, train_targets)
        st.markdown('#### Model Accuracy Score')
        st.write(f'**Accuracy: {acc*100:.2f}**')
    
    elif select_model == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=params['max_dept'], max_leaf_nodes=params['max_leaf_nodes'], random_state=1234) 
        acc = model_acc(train_inputs, train_targets)
        st.markdown('#### Model Accuracy Score')
        st.write(f'**Accuracy: {acc*100:.2f}**')

@st.cache
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

prediction_option = st.sidebar.selectbox('Make Prediction', ('No', 'Yes'))
try:
    if prediction_option == 'Yes':
        st.markdown('#### Making Prediction')
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
            st.write(f'Prediction: {prediction[0]}  \nProbability: {prediction[1]:.2f}')
except NameError:
    pass
