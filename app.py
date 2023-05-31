import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the preprocessed dataset
df = pd.read_csv('cancer_pred.csv')

#Loading the models
knn_model = pickle.load(open('models/knn_model.pkl', 'rb'))
dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))
nb_model = pickle.load(open('models/nb_model.pkl', 'rb'))
lr_model = pickle.load(open('models/lr_model.pkl', 'rb'))
sv_model = pickle.load(open('models/sv_model.pkl', 'rb'))

# Create a Streamlit app
def main():
    st.title('Breast Cancer Diagnosis Prediction')

    # Display user inputs for feature values
    st.header('Enter Feature Values')
    clump_thickness = st.selectbox('Clump Thickness', range(1, 11))
    cell_size_uniformity = st.selectbox('Uniformity of Cell Size', range(1, 11))
    cell_shape_uniformity = st.selectbox('Uniformity of Cell Shape', range(1, 11))
    marginal_adhesion = st.selectbox('Marginal Adhesion', range(1, 11))
    epithelial_cell_size = st.selectbox('Single Epithelial Cell Size', range(1, 11))
    bare_nuclei = st.selectbox('Bare Nuclei', range(1, 11))
    bland_chromatin = st.selectbox('Bland Chromatin', range(1, 11))
    normal_nucleoli = st.selectbox('Normal Nucleoli', range(1, 11))
    mitoses = st.selectbox('Mitoses', range(1, 11))

    # Creating a feature vector from the user inputs
    feature_vector = [[clump_thickness, cell_size_uniformity, cell_shape_uniformity, marginal_adhesion,
                       epithelial_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses]]

    st.title('Model Selection')
    model_name = st.selectbox('Select Model',
                              ['K-Nearest Neighbors', 'Decision Tree', 'Naive Bayes', 'Logistic Regression',
                               'Support Vector Machine'])

    # Creating a dictionary to map model names to loaded models
    models = {
        'K-Nearest Neighbors': knn_model,
        'Decision Tree': dt_model,
        'Naive Bayes': nb_model,
        'Logistic Regression': lr_model,
        'Support Vector Machine': sv_model
    }

    #prediction based on the selected model
    if st.button('Predict'):
        if model_name in models:
            selected_model = models[model_name]
            prediction = selected_model.predict(feature_vector)
            prediction = np.squeeze(prediction)  # Convert prediction array to 1D
            st.subheader('Prediction')
            st.write(f'The predicted diagnosis is: {prediction}')


if __name__ == '__main__':
    main()
