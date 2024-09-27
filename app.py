import streamlit as st
import xgboost as xgb
import pickle
from modules.text2vec import create_feature_row
from utils.ensemble_decision import prdiction

st.set_page_config(page_title='QuoraQP')

st.markdown(
    "<h3 style='font-size:34px;'>Quora Question Pair Detector - "
    "<a href='https://www.linkedin.com/in/prince-khunt-linked-in/' target='_blank'>Prince Khunt</a></h3>", 
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='font-size:24px;'>Enter ✌🏻 questions to see if both express the same idea.</h3>
    """, 
    unsafe_allow_html=True
)

question1 = st.text_input("Question 1:")
question2 = st.text_input("Question 2:")
result = None

with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# class CustomSpatialDropout1D(tf.keras.layers.SpatialDropout1D):
#     def __init__(self, rate, **kwargs):
#         kwargs.pop('trainable', None) 
#         super().__init__(rate, **kwargs)

# model = tf.keras.models.load_model('models/rnn_model.h5', custom_objects={'SpatialDropout1D': CustomSpatialDropout1D})

xgb_model = xgb.Booster()
xgb_model.load_model('models/xgboost_model.json')

if st.button('Check'):
    questions = [question1, question2]

    if question1 and question2:  
        feature_row = create_feature_row(question1, question2)
        feature_dmatrix = xgb.DMatrix(feature_row)

        rf_predict = rf_model.predict(feature_row)
        xgb_predict = xgb_model.predict(feature_dmatrix)

        if prdiction({'r': rf_predict, 'x': xgb_predict}):
            result = "👍🏻 Both questions express the same idea!" 
        else:
            result = "👎🏻 The questions express different ideas."

    else:
        result = "😠 Please enter both questions."

if result:
    st.markdown(f"<h4 style='font-size:20px;'>{result}</h4>", unsafe_allow_html=True)