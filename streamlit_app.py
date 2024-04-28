import pickle
import requests

# Function to load models from GitHub
def load_model(url):
    model_content = requests.get(url).content
    model = pickle.loads(model_content)
    return model

# Load machine learning models from GitHub
base_url = 'https://raw.githubusercontent.com/rohansatyarthi/finalproject/main/'
diabetes_model_kn_url = base_url + 'diabetes_model_kn.sav'
diabetes_model_lr_url = base_url + 'diabetes_model_lr.sav'
diabetes_model_rf_url = base_url + 'diabetes_model_rf.sav'
diabetes_model_svm_url = base_url + 'diabetes_model_svm.sav'

diabetes_model_kn = load_model(diabetes_model_kn_url)
diabetes_model_lr = load_model(diabetes_model_lr_url)
diabetes_model_rf = load_model(diabetes_model_rf_url)
diabetes_model_svm = load_model(diabetes_model_svm_url)

# Streamlit sidebar and main content...
with st.sidebar:
    selected = option_menu('Diabetes Prediction with different Algorithms',
                           ['Logistic Regression',
                            'K Nearest Neighbors',
                            'Random Forest',
                            'Support Vector Machine'],
                           default_index=0)

# Streamlit main content
st.title('Diabetes Prediction')

col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies')

with col2:
    Glucose = st.text_input('Glucose Level')

with col3:
    BloodPressure = st.text_input('Blood Pressure Value')

with col1:
    SkinThickness = st.text_input('Skin Thickness Value')

with col2:
    Insulin = st.text_input('Insulin Level')

with col3:
    BMI = st.text_input('BMI Value')

with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

with col2:
    Age = st.text_input('Age of the person')

if st.button('Diabetes Test Result'):
    results = []
    if selected == 'Logistic Regression':
        diab_prediction_lr = diabetes_model_lr.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age]])
        results.append(diab_prediction_lr[0])
    elif selected == 'K Nearest Neighbors':
        diab_prediction_kn = diabetes_model_kn.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age]])
        results.append(diab_prediction_kn[0])
    elif selected == 'Random Forest':
        diab_prediction_rf = diabetes_model_rf.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age]])
        results.append(diab_prediction_rf[0])
    elif selected == 'Support Vector Machine':
        diab_prediction_svm = diabetes_model_svm.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age]])
        results.append(diab_prediction_svm[0])

    average_result = sum(results) / len(results)
    if average_result >= 0.5:
        st.success('The average prediction across all algorithms indicates that the person is likely to be Diabetic.')
    else:
        st.success('The average prediction across all algorithms indicates that the person is likely not to be Diabetic.')