#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit.components.v1 import html  # 此导入保留但本版本未使用，可删除

# 设置页面配置
st.set_page_config(
    page_title="CLD Sarcopenia Risk Predictor",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Sarcopenia Risk Predictor")
st.markdown(
    """
    **For patients with Chronic Lung Disease (CLD)**  
    This tool predicts the probability of developing **sarcopenia**  
    based on four clinical features. Below the prediction, an interactive SHAP force plot  
    explains how each feature contributes to the individual prediction.
    """
)

# 加载模型、标准化器和 SHAP 解释器
@st.cache_resource
def load_artifacts():
    model = joblib.load('cld_sarcopenia_model.pkl')
    scaler = joblib.load('scaler.pkl')
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer

try:
    model, scaler, explainer = load_artifacts()
except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}. Ensure 'cld_sarcopenia_model.pkl' and 'scaler.pkl' are in the current directory.")
    st.stop()

# 侧边栏：输入参数（使用滑块）
st.sidebar.header("Patient Input")

age = st.sidebar.slider(
    "Age (years)",
    min_value=50, max_value=100, value=65, step=1,
    help="Age in years."
)

bmi = st.sidebar.slider(
    "BMI (kg/m²)",
    min_value=10.0, max_value=40.0, value=18.0, step=0.1,
    help="Body Mass Index (weight / height²)."
)

cognition = st.sidebar.slider(
    "Total Cognition Score",
    min_value=0.0, max_value=30.0, value=11.0, step=0.5,
    help="Higher score indicates better cognitive function."
)

disability = st.sidebar.selectbox(
    "Disability Status",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes",
    help="yes = disability, no = no disability."
)

predict_btn = st.sidebar.button("🔍 Predict & Explain", type="primary")

# 主区域：显示预测和 SHAP 图
if predict_btn:
    # 构造输入 DataFrame（列名必须与训练时一致）
    input_df = pd.DataFrame({
        'age': [age],
        'BMI': [bmi],                      # 训练时列名是 'BMI'
        'total_cognition': [cognition],
        'disability': [disability]
    })

    # 标准化
    input_scaled = scaler.transform(input_df)

    # 预测概率
    prob = model.predict_proba(input_scaled)[0, 1]

    # ---------- 预测结果显示 ----------
    st.subheader("📊 Prediction Result")
    col1, col2, col3 = st.columns(3)

    # 风险等级自定义阈值
    if prob < 0.017:
        risk_level = "Low"
        risk_message = "✅ Low risk, routine monitoring"
    elif prob < 0.07:
        risk_level = "Moderate"
        risk_message = "⚠️ Moderate risk, consider further assessment"
    else:
        risk_level = "High"
        risk_message = "🔴 High risk, comprehensive evaluation and intervention"

    with col1:
        st.metric("Sarcopenia Risk", f"{prob:.2%}")
    with col2:
        st.metric("Risk Level", risk_level, delta=None)
    with col3:
        # 根据风险等级显示对应建议
        if risk_level == "Low":
            st.success(risk_message)
        elif risk_level == "Moderate":
            st.warning(risk_message)
        else:
            st.error(risk_message)


    # 解释文本
    st.caption(
        f"""
        **Risk interpretation**  
        - **Low**    : Regular follow-up recommended.  
        - **Moderate**  : Clinical awareness and lifestyle intervention.  
        - **High**  : Comprehensive geriatric assessment and specialized care.
        """
    )

    # ---------- SHAP 力图（使用 matplotlib 静态图） ----------
    st.markdown("---")
    st.subheader("🔍 SHAP Force Plot ")

    input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(input_scaled_df)

    # 处理二分类情况：shap_values 可能是列表（正类索引为1）或数组
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # 取正类

    # 处理 expected_value：（取正类）
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]   # 取正类

    # 生成力图（返回 matplotlib figure）
    force_plot_fig = shap.force_plot(
        expected_value,
        shap_values[0, :],
        input_scaled_df.iloc[0, :],
        matplotlib=True,
        show=False
    )

    # 显示图形
    st.pyplot(force_plot_fig, bbox_inches='tight')

    # 额外显示输入回顾
    with st.expander("📋 Input Summary"):
        st.write(f"**Age**: {age} years")
        st.write(f"**BMI**: {bmi:.1f} kg/m²")
        st.write(f"**Total Cognition**: {cognition:.1f}")
        st.write(f"**Disability**: {'Yes' if disability==1 else 'No'}")

else:
    st.info("👈 Enter patient data in the sidebar and click **Predict & Explain**.")
    st.markdown(
        """
        **Features used in the model:**  
        - **Age** : years  
        - **BMI** : kg/m²  
        - **Total Cognition Score** : cognitive function  
        - **Disability** : Yes/No  
        """
    )

# -------------------------------
# 页脚
# -------------------------------
st.markdown("---")
st.caption("© 2026 CLD Sarcopenia Predictor | Model based on CHARLS data | SHAP for interpretability")
