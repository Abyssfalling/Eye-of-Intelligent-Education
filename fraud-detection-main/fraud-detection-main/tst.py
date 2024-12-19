from PIL import Image
# from streamlit_shap import st_shap
import streamlit as st
import numpy as np
import time
import plotly.express as px
import pandas as pd
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import shap
import catboost
from catboost import CatBoostClassifier
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.decomposition import PCA

plt.style.use('default')

st.set_page_config(
    page_title='Student Performance Management and Control',
    page_icon='🕵️‍♀️',
    layout='wide'
)

# dashboard title
# st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>学生成绩管控系统</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Student Performance Management and Control</h1>",
            unsafe_allow_html=True)


# side-bar
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.slider('Study Duration', -31.0, 3.0, 0.0)
    a2 = st.sidebar.slider('finished Task Num', -5.0, 13.0, 0.0)
    a3 = st.sidebar.slider('learned Lessons Num', -20.0, 6.0, 0.0)

    a4 = st.sidebar.slider('Note Num', -26.0, 7.0, 0.0)
    a5 = st.sidebar.slider('Discussion Participation', -4.0, 5.0, 0.0)

    a6 = st.sidebar.slider('joined Classroom Num', -8.0, 4.0, 0.0)
    a7 = st.sidebar.slider('joined CourseSet Num', 1.0, 5000.0, 1000.0)
    a8 = st.sidebar.slider('joined Course Num', 1.0, 5000.0, 1000.0)

    a9 = st.sidebar.selectbox("Gender?", ('Male', 'Female'))
    a10 = st.sidebar.selectbox("Role?", ('Student', 'Student&Assistant'))
    # 将a9转换为数值型
    if a9 == 'Male':
        a9_numeric = 1
    else:
        a9_numeric = 0

    # 将a10转换为数值型
    if a10 == 'Student':
        a10_numeric = 1
    else:
        a10_numeric = 0

    output = [a1, a2, a3, a4, a5, a6, a7, a8, a9_numeric, a10_numeric]
    # output = [a1, a2, a3, a4, a5, a7, a8, a9]
    return output


outputdf = user_input_features()

# understand the dataset
# df = pd.read_excel('"D:\database\\fraud-detection-main\\fraud-detection-main\student_info.xlsx"')
# df = pd.read_excel('D:\\database\\fraud-detection-main\\fraud-detection-main\\student_info.xlsx')
# df = pd.read_excel('D:\\database\\fraud-detection-main\\fraud-detection-main\\new2_student_info_filled.xlsx')
df = pd.read_excel('/share/users/gcsx/opt/fraud-detection-main/fraud-detection-main/new2_student_info_filled.xlsx')
df = df.sample(frac=0.1, random_state=42)

st.title('Dataset')
if st.button('View some random data'):
    st.write(df.sample(5))

st.write(f'The dataset is trained on Catboost with totally length of: {len(df)}. ')
# st.write(f'The dataset is trained on Catboost with totally length of: {len(df)}. 0️⃣ means its a real transaction, 1️⃣ means its a Fraud transaction. data is unbalanced (not⚖️)')

unbalancedf = pd.DataFrame(df.grade.value_counts())
st.write(unbalancedf)

# 需要一个count plot
placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()
placeholder4 = st.empty()