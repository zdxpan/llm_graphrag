import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 应用标题
# 
import streamlit as st

st.title('Hello, Streamlit!')

st.write('这是我的第一个Streamlit应用。')

import streamlit as st

# 文本输入
user_input = st.text_input("请输入您的名字", "John Doe")

# 滑动条
age = st.slider("请选择您的年龄", 18, 100, 30)

# 选择框
job = st.selectbox("请选择您的职业", ["学生", "教师", "工程师", "其他"])

st.write(f"Hello, {user_input}!")
st.write(f"您的年龄是 {age}，职业是 {job}。")


# 使用列进行布局
col1, col2, col3 = st.columns(3)
with col1:
    st.header("Column 1")
    st.write("这是第一列的内容")
with col2:
    st.header("Column 2")
    st.write("这是第二列的内容")
with col3:
    st.header("Column 3")
    st.write("这是第三列的内容")

# 使用展开器创建隐藏内容
with st.expander("点击展开更多信息"):
    st.write("这里是一些可以展开的详细信息。")

# 应用标题
st.title('交互式数据分析应用')

# 文件上传
uploaded_file = st.file_uploader("选择一个CSV文件", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)


import streamlit as st
import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    st.write(stringio)  # 不支持切片

    # To read file as string:
    string_data = stringio.read()
    st.write(stringio) # 不支持的

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)