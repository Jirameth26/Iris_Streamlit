import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. การตั้งค่าหน้าเพจแบบ Wide เพื่อให้เป็น Dashboard
st.set_page_config(page_title="Iris Predictor Dashboard", layout="wide")

# 2. โหลดโมเดล
@st.cache_resource
def load_model():
    # อ้างอิงไฟล์โมเดลที่บันทึกไว้
    return joblib.load('iris_model.pkl')

model = load_model()

# 3. --- ส่วนตั้งค่าแถบด้านข้าง (Sidebar) ---
st.sidebar.title("⚙️ Settings & Inputs")
st.sidebar.write("Tune the flower measurements to test the ML classifier model in real-time.")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0, 0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.0, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 3.0, 1.0, 0.1)

# 4. --- ส่วนการประมวลผลทำนาย ---
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]
species_names = ['Setosa', 'Versicolor', 'Virginica']

pred_species = species_names[prediction]
confidence = probabilities[prediction] * 100

# 5. --- ส่วนเนื้อหาหลัก (Main Layout) ---
st.title("Iris Flower Species Predictor 🌸")
st.markdown("An interactive ML dashboard to predict and compare flower dimensions.")
st.markdown("---")

# แบ่งหน้าจอเป็น 2 คอลัมน์ (ซ้าย และ ขวา)
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 🔮 Prediction Result")
    # สร้างกล่องสีสวยๆ แสดงผลทำนาย
    st.success(f"**PREDICTED SPECIES**\n# {pred_species}\n**Confidence:** {confidence:.1f}%")
    
    st.markdown("### Model Probability by Class")
    # กราฟแท่งแนวนอน (Horizontal Bar) แสดงความน่าจะเป็นของแต่ละสายพันธุ์
    fig_prob = go.Figure(go.Bar(
        x=probabilities * 100,
        y=species_names,
        orientation='h',
        marker_color=['#FF9999', '#66B2FF', '#99FF99'], # ปรับแต่งสีกราฟได้
        text=[f"{p*100:.1f}%" for p in probabilities],
        textposition='auto'
    ))
    fig_prob.update_layout(
        xaxis_title="Probability (%)", 
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
        height=250,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with col2:
    st.markdown("### 📊 Your Inputs vs Dataset Mean")
    st.markdown("Comparison of your configured flower dimensions against the overall Iris dataset mean.")
    
    # ค่าเฉลี่ยอ้างอิงของ Dataset (Sepal Length, Sepal Width, Petal Length, Petal Width)
    dataset_means = [5.84, 3.05, 3.76, 1.20]
    user_inputs = [sepal_length, sepal_width, petal_length, petal_width]
    feature_labels = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    # กราฟแท่งเปรียบเทียบ (Grouped Bar Chart)
    fig_comp = go.Figure(data=[
        go.Bar(name='Your Values', x=feature_labels, y=user_inputs, marker_color='#FF6692'),
        go.Bar(name='Dataset Average', x=feature_labels, y=dataset_means, marker_color='#B6E880')
    ])
    fig_comp.update_layout(
        barmode='group',
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp, use_container_width=True)