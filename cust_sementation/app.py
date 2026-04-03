import streamlit as st
import numpy as np
import joblib

# Page Config
st.set_page_config(page_title="Customer Segmentation", layout="centered")

# Load Model
@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

kmeans, scaler = load_model()

# Session State (IMPORTANT)
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "cluster" not in st.session_state:
    st.session_state.cluster = None

# Title
st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment")

# Input Form
with st.form("prediction_form"):
    
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Income", min_value=0.0, format="%.2f", help="Enter annual income ", key='income')
        age = st.number_input("Age", min_value=0,max_value=100, help="Enter age in years", key='age')
        spending = st.number_input("Total Spending", min_value=0.0, format="%.2f", help="Enter total spending amount", key='spending')

    with col2:
        recency = st.number_input("Recency", min_value=0, help="Enter number of days since last purchase", key='recency')
        children = st.number_input("Children", min_value=0, max_value=10, help="Enter number of children", key='children')
        purchases = st.number_input("Total Purchases", min_value=0.0, format="%.2f", help="Enter total number of purchases", key='purchases')

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        submit = st.form_submit_button("🔍 Predict")
    with col_btn2:
        clear = st.form_submit_button("🧹 Clear Result")

# Predict
if submit:
    data = np.array([[income, recency, age, children, spending, purchases]])
    data_scaled = scaler.transform(data)
    st.session_state.cluster = kmeans.predict(data_scaled)[0]
    st.session_state.show_result = True

# Clear ONLY result (not inputs)
if clear:
    st.session_state.show_result = False

# Show Result
if st.session_state.show_result:

    cluster = st.session_state.cluster

    st.markdown("---")
    st.subheader("Prediction Result")

    st.markdown(f"### 📊 Cluster Number: {cluster}")

    # Customer Type
    if cluster == 0:
        st.success("💎 High-Value Customer (Premium)")
    elif cluster == 1:
        st.info("🟢 Moderate / Family Customer")
    else:
        st.warning("🔴 Low-Value Customer")
    st.toast("Prediction Completed!")

    # METRIC CARDS 
    m1, m2, m3 = st.columns(3)
    m1.metric("Income", f"{income}")
    m2.metric("Spending", f"{spending}")
    m3.metric("Purchases", f"{purchases}")

    # INTERACTIVE SECTION (Tabs)
    tab1, tab2 = st.tabs(["🎁 Offers / Deals", "📢 Engagement Strategy"])

    # -------- OFFERS --------
    with tab1:

        if cluster == 0:
            with st.expander("View Offers"):
                st.write("• Exclusive VIP discounts (10–15%)")
                st.write("• Early access to new products")
                st.write("• Loyalty reward points / cashback")

        elif cluster == 1:
            with st.expander("View Offers"):
                st.write("• Bundle offers (Buy 2 Get 1 Free)")
                st.write("• Family packs / combo deals")
                st.write("• Festival or seasonal discounts")

        else:
            with st.expander("View Offers"):
                st.write("• Heavy discounts (20–30%)")
                st.write("• First purchase coupons")
                st.write("• Free delivery or cashback")

    # -------- ENGAGEMENT --------
    with tab2:

        if cluster == 0:
            with st.expander("View Strategy"):
                st.write("• Personalized recommendations")
                st.write("• Premium membership / subscription plans")
                st.write("• Invite to special events or sales")

        elif cluster == 1:
            with st.expander("View Strategy"):
                st.write("• Targeted email/SMS with offers")
                st.write("• Discounts on essential products")
                st.write("• Coupons for repeat purchases")

        else:
            with st.expander("View Strategy"):
                st.write("• Retargeting ads")
                st.write("• Push notifications")
                st.write("• Limited-time offers (urgency)")

    # Goal Section
    st.markdown("### 🎯 Business Goal")

    if cluster == 0:
        st.success("Retain customers and increase loyalty")
    elif cluster == 1:
        st.info("Increase spending and convert to high-value customers")
    else:
        st.warning("Increase engagement and encourage more purchases")



