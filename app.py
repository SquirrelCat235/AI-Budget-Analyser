import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Configuration
st.set_page_config(page_title="AI Budget Analyser", page_icon="💸", layout="wide")

st.markdown("""
<style>
/* Add some standout styling */
.big-font {
    font-size:30px !important;
    font-weight: bold;
    margin-bottom: 0px;
}
.metric-card {
    background-color: #2E3440;
    color: #ECEFF4;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    text-align: center;
    border: 1px solid #4C566A;
}
.metric-card h3 {
    margin-top: 0px;
    color: #8FBCBB;
    font-size: 18px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

st.title("💸 AI-Powered Budget Analyzer & Predictor")
st.markdown("Analyze your spending patterns, predict next month's expenses, and get personalized financial recommendations.")

# Function to generate dummy data
@st.cache_data
def load_dummy_data():
    np.random.seed(42)
    categories = ['Groceries', 'Rent', 'Utilities', 'Entertainment', 'Dining Out', 'Transportation', 'Healthcare', 'Miscellaneous']
    
    data = []
    start_date = datetime(datetime.now().year, 1, 1)
    end_date = datetime.now()
    days_to_generate = (end_date - start_date).days
    
    for i in range(days_to_generate + 1):
        current_date = start_date + timedelta(days=i)
        # Add random expenses for each day
        num_transactions = np.random.randint(1, 5)
        for _ in range(num_transactions):
            cat = np.random.choice(categories, p=[0.2, 0.05, 0.1, 0.15, 0.2, 0.1, 0.05, 0.15])
            
            # Base amount depending on category
            if cat == 'Rent':
                amount = 20000 if current_date.day == 1 else 0
            elif cat == 'Utilities':
                amount = np.random.uniform(1500, 3500) if current_date.day == 5 else 0
            else:
                amount = np.random.uniform(200, 1500)
            
            if amount > 0:
                data.append({
                    'Date': current_date,
                    'Category': cat,
                    'Amount': round(amount, 2)
                })
    
    df = pd.DataFrame(data)
    # Give it an upward trend over time
    df['Amount'] = df['Amount'] * (1 + (df['Date'] - start_date).dt.days / max(1, days_to_generate) * 0.1)
    return df

st.sidebar.header("Upload your data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Category, Amount)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
else:
    st.sidebar.info("No CSV uploaded. Demonstrating with dummy data.")
    df = load_dummy_data()

# Ensure standard columns
if not all(col in df.columns for col in ['Date', 'Category', 'Amount']):
    st.error("Uploaded CSV must contain 'Date', 'Category', and 'Amount' columns.")
    st.stop()

df['Month'] = df['Date'].dt.to_period('M').astype(str)

# --- Filters ---
st.sidebar.header("Filters")
selected_month = st.sidebar.selectbox("Select Month for Analysis", options=sorted(df['Month'].unique(), reverse=True))

# --- Data Processing ---
current_month_df = df[df['Month'] == selected_month]
total_spent = current_month_df['Amount'].sum()
category_breakdown = current_month_df.groupby('Category')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False)
top_category = category_breakdown.iloc[0]['Category'] if not category_breakdown.empty else "N/A"
top_category_amount = category_breakdown.iloc[0]['Amount'] if not category_breakdown.empty else 0

# --- Dashboard ---
st.header(f"📊 Historical Dashboard - {selected_month}")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'<div class="metric-card"><h3>Total Spent</h3><p class="big-font">₹{total_spent:,.2f}</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><h3>Top Category</h3><p class="big-font">{top_category}</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><h3>Top Spend</h3><p class="big-font">₹{top_category_amount:,.2f}</p></div>', unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Spending by Category")
    if not category_breakdown.empty:
        fig_pie = px.pie(category_breakdown, values='Amount', names='Category', hole=0.5, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ECEFF4")
        st.plotly_chart(fig_pie, width='stretch')
    else:
        st.info("No data for pie chart.")

with col2:
    st.subheader("Daily Spending Over Time")
    daily_spending = current_month_df.groupby(current_month_df['Date'].dt.date)['Amount'].sum().reset_index()
    if not daily_spending.empty:
        fig_line = px.line(daily_spending, x='Date', y='Amount', markers=True, 
                           line_shape='spline', color_discrete_sequence=['#88C0D0'])
        fig_line.update_layout(margin=dict(t=0, b=0, l=0, r=0), xaxis_title="Date", yaxis_title="Amount (₹)",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ECEFF4")
        st.plotly_chart(fig_line, width='stretch')
    else:
        st.info("No data for line chart.")

# --- Prediction Model ---
st.markdown("---")

# Determine what the "next month" is for display purposes
if not df.empty:
    latest_month = pd.to_datetime(df['Month'].max())
    next_month_name = (latest_month + pd.DateOffset(months=1)).strftime('%B %Y')
else:
    next_month_name = "Next Month"

st.header(f"🔮 Expense Prediction for {next_month_name}")

st.sidebar.markdown("---")
st.sidebar.header("Prediction Settings")
num_months_pred = st.sidebar.slider("Months of history for prediction", min_value=3, max_value=12, value=min(6, len(df['Month'].unique())))

# Group data by month to build a model
monthly_totals = df.groupby('Month')['Amount'].sum().reset_index()
monthly_totals['Month_Date'] = pd.to_datetime(monthly_totals['Month'])
monthly_totals = monthly_totals.sort_values('Month_Date')

# Filter to include only the specified number of previous months for prediction
if len(monthly_totals) > num_months_pred:
    monthly_totals = monthly_totals.tail(num_months_pred)

monthly_totals['Month_Index'] = np.arange(len(monthly_totals))

if len(monthly_totals) >= 3:
    # Linear Regression for simple trend prediction
    X = monthly_totals[['Month_Index']]
    y = monthly_totals['Amount']

    model = LinearRegression()
    model.fit(X, y)

    # Predict next month
    next_index = monthly_totals['Month_Index'].max() + 1
    predicted_amount = model.predict([[next_index]])[0]

    # Predict next month per category for more detailed insights
    cat_predictions = {}
    for cat in df['Category'].unique():
        cat_data = df[df['Category'] == cat].groupby('Month')['Amount'].sum().reset_index()
        # Merge with all months to handle missing months with 0
        cat_data = pd.merge(monthly_totals[['Month', 'Month_Index']], cat_data, on='Month', how='left').fillna({'Amount': 0})
        
        if len(cat_data) >= 3:
            model_cat = LinearRegression()
            model_cat.fit(cat_data[['Month_Index']], cat_data['Amount'])
            cat_pred = model_cat.predict([[next_index]])[0]
            cat_predictions[cat] = max(0, cat_pred) # Prevent negative predictions

    col_pred1, col_pred2 = st.columns([1, 2])
    
    with col_pred1:
        st.info(f"Based on your past spending, we predict your {next_month_name} expenses will be:")
        st.markdown(f'<h1 style="color:#A3BE8C; text-align:center;">₹{predicted_amount:,.2f}</h1>', unsafe_allow_html=True)
        
        trend = "increasing" if model.coef_[0] > 0 else "decreasing"
        st.write(f"Your spending trend is **{trend}** by about ₹{abs(model.coef_[0]):,.2f} per month.")
        st.write("Keep this budget target in mind to improve your savings rate.")
        
    with col_pred2:
        # Show historical + predicted
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=monthly_totals['Month'], y=monthly_totals['Amount'], 
                                      mode='lines+markers', name='Actual', line={"color": '#81A1C1', "width": 3}))
        
        # Add prediction point
        next_month_str = (monthly_totals['Month_Date'].max() + pd.DateOffset(months=1)).strftime('%Y-%m')
        fig_pred.add_trace(go.Scatter(x=[monthly_totals['Month'].iloc[-1], next_month_str], 
                                      y=[monthly_totals['Amount'].iloc[-1], predicted_amount], 
                                      mode='lines+markers', name='Predicted (Trend)', line={"color": '#BF616A', "width": 3, "dash": 'dash'}))
        
        fig_pred.update_layout(title="Historical vs Predicted Monthly Spending", xaxis_title="Month", yaxis_title="Total Amount (₹)",
                               margin=dict(t=40, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ECEFF4")
        st.plotly_chart(fig_pred, width='stretch')
else:
    st.warning("Not enough data to make predictions. Need at least 3 months of data.")

# --- Recommendations ---
st.markdown("---")
st.header("💡 AI Financial Recommendations")

if not current_month_df.empty:
    recommendations = []
    
    # 1. High spending category alert
    if top_category_amount > total_spent * 0.4 and top_category != 'Rent':
         recommendations.append(f"⚠️ Your spending on **{top_category}** makes up {top_category_amount/total_spent*100:.1f}% of your budget this month. Consider setting a strict limit for this category.")
    
    # 2. Dining out vs Groceries
    dining = current_month_df[current_month_df['Category'] == 'Dining Out']['Amount'].sum() if 'Dining Out' in current_month_df['Category'].values else 0
    groceries = current_month_df[current_month_df['Category'] == 'Groceries']['Amount'].sum() if 'Groceries' in current_month_df['Category'].values else 0
    
    if dining > groceries and dining > 0:
        recommendations.append("🍔 You spent more on **Dining Out** than **Groceries** this month. Cooking at home could save you a significant amount next month.")
    
    # 3. Overall trend warning
    if len(monthly_totals) >= 2:
        last_month_spent = monthly_totals.iloc[-2]['Amount']
        if total_spent > last_month_spent * 1.1:
            recommendations.append(f"📈 Your spending this month is up by >10% compared to last month. Watch out for lifestyle inflation!")
        elif total_spent < last_month_spent * 0.9:
            recommendations.append(f"🎉 Great job! Your spending is down by more than 10% compared to last month.")
            
    # 4. Predict categories that might overspend next month
    if len(monthly_totals) >= 3 and cat_predictions:
        predicted_top_cat, predicted_top_val = max(cat_predictions.items(), key=lambda x: x[1])
        # Compare prediction vs average of last 3 months
        cat_data_top = df[df['Category'] == predicted_top_cat].groupby('Month')['Amount'].sum().tail(3).mean()
        if predicted_top_cat != 'Rent' and predicted_top_val > cat_data_top * 1.15:
             recommendations.append(f"🔮 Our model predicts that in {next_month_name} your **{predicted_top_cat}** expenses might rise significantly to ~₹{predicted_top_val:,.2f}. Try to cut back early!")

    if not recommendations:
        recommendations.append("✅ Your spending looks balanced this month. Keep up the good work saving!")
        
    for i, rec in enumerate(recommendations, 1):
        st.info(f"**{i}.** {rec}")

else:
    st.info("No data available for the current month to generate recommendations.")

# Footer
st.markdown("---")
st.markdown("### 📝 About")
st.markdown("This standout dashboard is built with **Python**, **Streamlit**, **Scikit-learn** and **Plotly**.")
