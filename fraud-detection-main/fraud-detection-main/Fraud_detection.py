# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import numpy as np
import time
import plotly.express as px
import pandas as pd
import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
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
    page_title = 'Student Performance Management and Control',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>学生成绩管控系统</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Student Performance Management and Control</h1>", unsafe_allow_html=True)

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
    a10 = st.sidebar.selectbox("Role?", ('Student','Student&Assistant'))
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

    output = [a1,a2,a3,a4,a5,a6,a7,a8,a9_numeric,a10_numeric]
    #output = [a1, a2, a3, a4, a5, a7, a8, a9]
    return output

outputdf = user_input_features()



# understand the dataset
#df = pd.read_excel('"D:\database\\fraud-detection-main\\fraud-detection-main\student_info.xlsx"')
#df = pd.read_excel('D:\\database\\fraud-detection-main\\fraud-detection-main\\student_info.xlsx')
#df = pd.read_excel('D:\\database\\fraud-detection-main\\fraud-detection-main\\new2_student_info_filled.xlsx')
df = pd.read_excel('/share/users/gcsx/opt/fraud-detection-main/fraud-detection-main/new2_student_info_filled.xlsx')
df=df.sample(frac=0.1, random_state=42)

st.title('Dataset')
if st.button('View some random data'):
    st.write(df.sample(5))
    
st.write(f'The dataset is trained on Catboost with totally length of: {len(df)}. ')
#st.write(f'The dataset is trained on Catboost with totally length of: {len(df)}. 0️⃣ means its a real transaction, 1️⃣ means its a Fraud transaction. data is unbalanced (not⚖️)')

unbalancedf = pd.DataFrame(df.grade.value_counts())
st.write(unbalancedf)



# 需要一个count plot
placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()
placeholder4 = st.empty()


with placeholder.container():
    f1,f2,f3 = st.columns(3)

    with f1:
        a1A = df[df['grade'] == 'A']['learnedSeconds']
        a1B = df[df['grade'] == 'B']['learnedSeconds']
        a1C = df[df['grade'] == 'C']['learnedSeconds']
        a1D = df[df['grade'] == 'D']['learnedSeconds']
        a1E = df[df['grade'] == 'E']['learnedSeconds']
        a1F = df[df['grade'] == 'F']['learnedSeconds']
        # a1G = df[df['grade'] == 'G']['learnedSeconds']
        # a1H = df[df['grade'] == 'H']['learnedSeconds']
        hist_data = [a1A,a1B,a1C,a1D,a1E,a1F]
        #group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data,group_labels = ['A', 'B','C', 'D','E', 'F'])
        fig.update_layout(title_text='learnedSeconds')
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        a2A = df[df['grade'] == 'A']['finishedTaskNum']
        a2B = df[df['grade'] == 'B']['finishedTaskNum']
        a2C = df[df['grade'] == 'C']['finishedTaskNum']
        a2D = df[df['grade'] == 'D']['finishedTaskNum']
        a2E = df[df['grade'] == 'E']['finishedTaskNum']
        a2F = df[df['grade'] == 'F']['finishedTaskNum']
        # a2G = df[df['grade'] == 'G']['finishedTaskNum']
        # a2H = df[df['grade'] == 'H']['finishedTaskNum']
        hist_data = [a2A, a2B, a2C, a2D, a2E, a2F]
        #group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels=['A', 'B', 'C', 'D', 'E', 'F'])
        fig.update_layout(title_text='finishedTaskNum')
        st.plotly_chart(fig, use_container_width=True)
    with f3:
        a3A = df[df['grade'] == 'A']['learnedNum']
        a3B = df[df['grade'] == 'B']['learnedNum']
        a3C = df[df['grade'] == 'C']['learnedNum']
        a3D = df[df['grade'] == 'D']['learnedNum']
        a3E = df[df['grade'] == 'E']['learnedNum']
        a3F = df[df['grade'] == 'F']['learnedNum']
        # a3G = df[df['grade'] == 'G']['learnedNum']
        # a3H = df[df['grade'] == 'H']['learnedNum']
        hist_data = [a3A, a3B, a3C, a3D, a3E, a3F]
        fig = ff.create_distplot(hist_data, group_labels=['A', 'B', 'C', 'D', 'E', 'F'])
        fig.update_layout(title_text='learnedNum')
        st.plotly_chart(fig, use_container_width=True)


with placeholder2.container():
    f4,f5 = st.columns(2)

    with f4:
        a4A = df[df['grade'] == 'A']['noteNum']
        a4B = df[df['grade'] == 'B']['noteNum']
        a4C = df[df['grade'] == 'C']['noteNum']
        a4D = df[df['grade'] == 'D']['noteNum']
        a4E = df[df['grade'] == 'E']['noteNum']
        a4F = df[df['grade'] == 'F']['noteNum']
        # a4G = df[df['grade'] == 'G']['noteNum']
        # a4H = df[df['grade'] == 'H']['noteNum']
        hist_data = [a4A, a4B, a4C, a4D, a4E, a4F]
        # 使用主成分分析进行数据降维
        # pca = PCA(n_components=1)  # 设置你想要的维度
        # hist_data_pca = [pca.fit_transform(np.array(data).reshape(-1, 1)) for data in hist_data]

        # 创建分布图
        #fig = ff.create_distplot(hist_data_pca, group_labels=['A', 'B', 'C', 'D', 'E', 'F'])
        fig = ff.create_distplot(hist_data, group_labels=['A', 'B', 'C', 'D', 'E', 'F'])
        fig.update_layout(title_text='noteNum')
        st.plotly_chart(fig, use_container_width=True)
    with f5:
        a5A = df[df['grade'] == 'A']['threadNum']
        a5B = df[df['grade'] == 'B']['threadNum']
        a5C = df[df['grade'] == 'C']['threadNum']
        a5D = df[df['grade'] == 'D']['threadNum']
        a5E = df[df['grade'] == 'E']['threadNum']
        a5F = df[df['grade'] == 'F']['threadNum']
        # a5G = df[df['grade'] == 'G']['threadNum']
        # a5H = df[df['grade'] == 'H']['threadNum']
        hist_data = [a5A, a5B, a5C, a5D, a5E, a5F]
        fig = ff.create_distplot(hist_data, group_labels=['A', 'B', 'C', 'D', 'E', 'F' ])
        fig.update_layout(title_text='threadNum')
        st.plotly_chart(fig, use_container_width=True)


with placeholder3.container():
    f6,f7,f8 = st.columns(3)

    with f6:
        a6A = df[df['grade'] == 'A']['joinedClassroomNum']
        a6B = df[df['grade'] == 'B']['joinedClassroomNum']
        a6C = df[df['grade'] == 'C']['joinedClassroomNum']
        a6D = df[df['grade'] == 'D']['joinedClassroomNum']
        a6E = df[df['grade'] == 'E']['joinedClassroomNum']
        a6F = df[df['grade'] == 'F']['joinedClassroomNum']
        # a6G = df[df['grade'] == 'G']['joinedClassroomNum']
        # a6H = df[df['grade'] == 'H']['joinedClassroomNum']
        hist_data = [a6A, a6B, a6C, a6D, a6E, a6F]
        fig = ff.create_distplot(hist_data, group_labels=['A', 'B', 'C', 'D', 'E', 'F'])
        fig.update_layout(title_text='joinedClassroomNum')
        st.plotly_chart(fig, use_container_width=True)
    with f7:
        a7A = df[df['grade'] == 'A']['joinedCourseSetNum']
        a7B = df[df['grade'] == 'B']['joinedCourseSetNum']
        a7C = df[df['grade'] == 'C']['joinedCourseSetNum']
        a7D = df[df['grade'] == 'D']['joinedCourseSetNum']
        a7E = df[df['grade'] == 'E']['joinedCourseSetNum']
        a7F = df[df['grade'] == 'F']['joinedCourseSetNum']
        # a7G = df[df['grade'] == 'G']['joinedCourseSetNum']
        # a7H = df[df['grade'] == 'H']['joinedCourseSetNum']
        hist_data = [a7A, a7B, a7C, a7D, a7E, a7F]
        fig = ff.create_distplot(hist_data, group_labels=['A', 'B', 'C', 'D', 'E', 'F'])
        fig.update_layout(title_text='joinedCourseSetNum')
        st.plotly_chart(fig, use_container_width=True)
    with f8:
        a8A = df[df['grade'] == 'A']['joinedCourseNum']
        a8B = df[df['grade'] == 'B']['joinedCourseNum']
        a8C = df[df['grade'] == 'C']['joinedCourseNum']
        a8D = df[df['grade'] == 'D']['joinedCourseNum']
        a8E = df[df['grade'] == 'E']['joinedCourseNum']
        a8F = df[df['grade'] == 'F']['joinedCourseNum']
        # a8G = df[df['grade'] == 'G']['joinedCourseNum']
        # a8H = df[df['grade'] == 'H']['joinedCourseNum']
        hist_data = [a8A, a8B, a8C, a8D, a8E, a8F]
        fig = ff.create_distplot(hist_data, group_labels=['A', 'B', 'C', 'D', 'E', 'F'])
        fig.update_layout(title_text='joinedCourseNum')
        st.plotly_chart(fig, use_container_width=True)

df2 = df[['grade','Gender']].value_counts().reset_index()

df3 = df[['grade','role']].value_counts().reset_index()



with placeholder4.container():
    f1,f2 = st.columns(2)


    with f1:
        #fig = plt.figure()
        fig = px.bar(df2, x='grade', y='Gender', color='Gender', color_continuous_scale=px.colors.qualitative.Plotly,  title=" Gender: 🔴Female; 🔵Male")
        st.write(fig)        

    with f3:
        fig = px.bar(df3, x='grade', y= 'role', color="role", title="Role:  ❤️Student; 💙Student|Assistant")
        st.write(fig)







st.title('SHAP Value')

image4 = Image.open(r'/share/users/gcsx/opt/fraud-detection-main/fraud-detection-main/summary.png')
shapdatadf =pd.read_excel(r'/share/users/gcsx/opt/fraud-detection-main/fraud-detection-main/shapdatadf.xlsx')
shapvaluedf =pd.read_excel(r'/share/users/gcsx/opt/fraud-detection-main/fraud-detection-main/shapvaluedf.xlsx')





placeholder5 = st.empty()
with placeholder5.container():
    f1,f2 = st.columns(2)

    with f1:
        st.subheader('Summary plot')
        st.write('👈 Lower')
        st.write('👉 Higher')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.image(image4)     
    # with f2:
    #     st.subheader('Dependence plot for features')
    #     cf = st.selectbox("Choose a feature", (shapdatadf.columns))
    #
    #
    #     fig = px.scatter(x = shapdatadf[cf],
    #                      y = shapvaluedf[cf],
    #                      color=shapdatadf[cf],
    #                      color_continuous_scale= ['blue','red'],
    #                      labels={'x':'Original value', 'y':'shap value'})
    #     st.write(fig)

