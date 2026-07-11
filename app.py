import streamlit as st
import pandas as pd
import joblib
import datetime

# --- 1. Page Config ---
st.set_page_config(page_title="AI Sales Analytics", layout="wide")
st.title("📊 AI Sales & Worker Performance Dashboard")

# --- 2. Load Data and Models (Module 1: Sales Prediction) ---
@st.cache_resource
def load_assets():
    model = joblib.load('final_m1_model.pkl')
    office_encoder = joblib.load('office_encoder.pkl')
    product_encoder = joblib.load('product_encoder.pkl')
    data = pd.read_csv('final_m1_data.csv')
    data['Sale Date'] = pd.to_datetime(data['Sale Date'])
    return model, office_encoder, product_encoder, data

model, office_encoder, product_encoder, data = load_assets()

all_offices = data['Office Name'].unique()
all_products = data['Product'].unique()

# --- Helper Function for Macro Predictions ---
def predict_date_range(start_date, end_date, office_id, product_id):
    dates = pd.date_range(start=start_date, end=end_date)
    input_data = pd.DataFrame({
        'Day': dates.day,
        'Month': dates.month,
        'DayOfWeek': dates.weekday,
        'Office_ID': office_id,
        'Product_ID': product_id
    })
    predictions = model.predict(input_data)
    total_sales = sum([max(0, int(p)) for p in predictions])
    return total_sales

# --- 3. Top-Level Tabs ---
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs(["Sales Prediction","Demand Forecasting", "Worker Performance", "Early Warning System","Product Recommendation", "Sales Trends"])

# ============================================================
# MAIN TAB 1: SALES PREDICTION (Micro + Macro)
# ============================================================
with main_tab1:
    st.write("Forecast demand at both the micro (daily) and macro (strategic) levels.")

    # Global Filters
    st.write("### ⚙️ Select Parameters")
    col1, col2 = st.columns(2)
    with col1:
        selected_office = st.selectbox("📍 Select Office:", all_offices, key="sales_office")
    with col2:
        selected_product = st.selectbox("📦 Select Product:", all_products, key="sales_product")

    office_id = office_encoder.transform([selected_office])[0]
    product_id = product_encoder.transform([selected_product])[0]

    # Historical Sales Chart
    st.write(f"### 📊 Historical Sales: {selected_product} at {selected_office}")
    filtered_data = data[(data['Office Name'] == selected_office) & (data['Product'] == selected_product)]
    if not filtered_data.empty:
        st.line_chart(filtered_data.set_index('Sale Date')['Qty'])
    else:
        st.info("No historical data available for this specific combination.")

    st.divider()

    # Sub-tabs for Micro and Macro predictions
    sub_tab1, sub_tab2 = st.tabs(["📅 Daily Prediction (Micro)", "📈 Strategic Forecast (Macro)"])

    # ---- Sub Tab 1: Micro (Daily) ----
    with sub_tab1:
        st.write("### 🔮 Predict Daily Demand")
        selected_date = st.date_input("Select a specific future date:", datetime.date(2026, 8, 1))

        if st.button("Predict Daily Sales"):
            day = selected_date.day
            month = selected_date.month
            dayofweek = selected_date.weekday()

            input_data = pd.DataFrame([[day, month, dayofweek, office_id, product_id]],
                                       columns=['Day', 'Month', 'DayOfWeek', 'Office_ID', 'Product_ID'])

            prediction = model.predict(input_data)
            predicted_qty = max(0, int(prediction[0]))

            st.success(f"**Predicted demand for {selected_product} at {selected_office} on {selected_date}:** {predicted_qty} Units")

    # ---- Sub Tab 2: Macro (Strategic) ----
    with sub_tab2:
        st.write("### 📊 Generate Strategic Forecast Reports")
        target_year = 2026

        if st.button("Generate Macro Forecast"):
            with st.spinner("Calculating high-level predictions..."):

                aug_start, aug_end = datetime.date(target_year, 8, 1), datetime.date(target_year, 8, 31)
                next_month_qty = predict_date_range(aug_start, aug_end, office_id, product_id)

                q4_start, q4_end = datetime.date(target_year, 10, 1), datetime.date(target_year, 12, 31)
                next_quarter_qty = predict_date_range(q4_start, q4_end, office_id, product_id)

                fest_start, fest_end = datetime.date(target_year, 9, 15), datetime.date(target_year, 11, 15)
                fest_qty = predict_date_range(fest_start, fest_end, office_id, product_id)

                annual_start, annual_end = datetime.date(target_year + 1, 1, 1), datetime.date(target_year + 1, 12, 31)
                annual_qty = predict_date_range(annual_start, annual_end, office_id, product_id)

            st.success("Strategic Forecast Generated!")

            metric1, metric2 = st.columns(2)
            metric1.metric(label="Next Month Sales (Aug '26)", value=f"{next_month_qty} units")
            metric2.metric(label="Next Quarter Demand (Q4 '26)", value=f"{next_quarter_qty} units")

            metric3, metric4 = st.columns(2)
            metric3.metric(label="Festival Season Demand", value=f"{fest_qty} units")
            metric4.metric(label="Annual Forecast (2027)", value=f"{annual_qty} units")

# ============================================================
# MAIN TAB 3: WORKER PERFORMANCE
# ============================================================
with main_tab3:
    st.header("Worker Performance & Risk Analytics")
    st.metric(
    "Model Accuracy",
    "95.4%"
)
    try:
        worker_df = pd.read_csv('worker_performance_report.csv')

        # Filter by office (separate key to avoid clashing with sales office selector)
        office_list = ["All Offices"] + list(worker_df['Office Name'].unique())
        selected_worker_office = st.selectbox("Filter by Office:", office_list, key="worker_office")

        if selected_worker_office != "All Offices":
            worker_df = worker_df[worker_df['Office Name'] == selected_worker_office]

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Workers", len(worker_df))
        col2.metric("High Performers", len(worker_df[worker_df['Risk_Category'] == 'Safe (High Performer)']))
        col3.metric("High Risk Workers", len(worker_df[worker_df['Risk_Category'] == 'High Risk (Underperforming)']))

        # Interactive table
        st.write("### Worker Ranking Table")
        st.dataframe(
            worker_df.style.map(
                lambda x: 'background-color: #ffcccc' if x == 'High Risk (Underperforming)'
                else ('background-color: #ccffcc' if x == 'Safe (High Performer)' else ''),
                subset=['Risk_Category']
            ),
            use_container_width=True
        )

    except FileNotFoundError:
        st.error("Could not find 'worker_performance_report.csv'. Make sure you ran Step 1 in Jupyter!")
# ============================================================
# MAIN TAB 2: Demand Forecasting
# ============================================================
with main_tab2:

    st.header("📦 AI Procurement Planning Dashboard")
    st.write("Forecast next month's demand and generate procurement recommendations.")

    # Select Office and Product
    col1, col2 = st.columns(2)

    with col1:
        proc_office = st.selectbox(
            "📍 Office",
            all_offices,
            key="proc_office"
        )

    with col2:
        proc_product = st.selectbox(
            "📦 Product",
            all_products,
            key="proc_product"
        )

    # Stock and Cost
    col3, col4 = st.columns(2)

    with col3:
        current_stock = st.number_input(
            "Current Stock",
            min_value=0,
            value=250
        )

    with col4:
        cost_per_unit = st.number_input(
            "Cost per Unit (₹)",
            min_value=0.0,
            value=50.0
        )

    if st.button("Generate Procurement Plan"):

        office_id = office_encoder.transform([proc_office])[0]
        product_id = product_encoder.transform([proc_product])[0]

        # Next Month
        start_date = datetime.date(2026, 8, 1)
        end_date = datetime.date(2026, 8, 31)

        predicted_demand = predict_date_range(
            start_date,
            end_date,
            office_id,
            product_id
        )

        procure_qty = max(
            0,
            predicted_demand - current_stock
        )

        procurement_cost = procure_qty * cost_per_unit

        if predicted_demand > 0:
            coverage = (current_stock / predicted_demand) * 100
        else:
            coverage = 100

        if coverage >= 120:
            status = "🟢 Overstock"

        elif coverage >= 80:
            status = "🟢 Healthy"

        elif coverage >= 50:
            status = "🟠 Low Stock"

        else:
            status = "🔴 Critical"

        st.divider()

        st.subheader("📊 Procurement Report")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Current Stock",
            f"{current_stock:,}"
        )

        c2.metric(
            "Forecast Demand",
            f"{predicted_demand:,}"
        )

        c3.metric(
            "Need to Procure",
            f"{procure_qty:,}"
        )

        st.progress(min(coverage / 100, 1.0))

        st.write(f"### Inventory Coverage : {coverage:.1f}%")

        st.write(f"### Inventory Status : {status}")

        st.metric(
            "Estimated Procurement Cost",
            f"₹{procurement_cost:,.2f}"
        )

        if procure_qty > 0:

            st.warning(
                f"""
Procure **{procure_qty} units** of **{proc_product}**
for **{proc_office}** before next month.
"""
            )

        else:

            st.success(
                "Current inventory is sufficient for the predicted demand."
            )
with main_tab4:
    st.header("🚨 Module 4: AI Early Warning System")
    st.write("Identifies workers likely to miss their monthly targets based on behavioral patterns (Working Days and Quantity).")
    
    try:
        # Load the predictions you exported from Jupyter
        df_m4 = pd.read_csv("m4_worker_predictions.csv")
        
        # Split the dashboard into two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚠️ Action Required")
            # Filter for workers who failed to meet the 60% threshold
            at_risk_workers = df_m4[df_m4['Risk Category'] == 'Needs Monitoring']
            
            st.metric("Total Workers at Risk", len(at_risk_workers))
            
            st.write("**High-Risk Roster (Lowest Probability First)**")
            # Display just the names and percentages of the at-risk workers, sorted lowest to highest
            st.dataframe(
                at_risk_workers[['Worker Name', 'Target_Probability_%']].sort_values('Target_Probability_%'),
                hide_index=True,
                use_container_width=True
            )
            
        with col2:
            st.subheader("📋 All Worker Predictions")
            
            # Create a simple color-coding function for Streamlit
            def highlight_risk(val):
                if val == 'Needs Monitoring':
                    return 'background-color: #ffcccc; color: #900000;' # Light Red
                elif val == 'On Track':
                    return 'background-color: #ccffcc; color: #006600;' # Light Green
                return ''
                
            # Apply the color coding to the 'Risk Category' column
            styled_df = df_m4.style.map(highlight_risk, subset=['Risk Category'])
            
            # Display the beautifully formatted table
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
            
    except FileNotFoundError:
        st.warning("⚠️ Waiting for data. Please run your Jupyter Notebook to generate 'm4_worker_predictions.csv'.")
with main_tab5:
    st.header("💡 Module 5: AI Product Recommendation")
    st.write("AI-driven suggestions for product supply based on office demand patterns.")

    try:
        # Load the recommendations file
        df_recs = pd.read_csv("m5_product_recommendations.csv")

        # 1. Dropdown for selecting an office
        st.subheader("Select Office to View Recommendations")
        selected_office = st.selectbox(
            "Which office do you need to stock?", 
            options=df_recs['Office Name'].unique(),
            key='m5_office_select'
        )

        # 2. Display the recommendation
        # Filter the dataframe for the selected office
        office_rec = df_recs[df_recs['Office Name'] == selected_office]['Recommended Products'].iloc[0]

        # Use an attractive layout to show the products
        st.success(f"### Recommended Products for {selected_office}")
        st.write(f"Based on historical sales data, prioritize the supply of: **{office_rec}**")
        
        # 3. Visual insight
        st.info("💡 Note: These recommendations are based on the highest sales volume recorded for each product at this location.")

        # 4. View all table
        with st.expander("View Full Recommendation Matrix"):
            st.dataframe(df_recs, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ Module 5 data not found. Please run your Jupyter code to generate 'm5_product_recommendations.csv'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
with main_tab6: # Update this variable to match your new tab name (e.g., tab5)
    st.header("📈 Module 6: Sales Trend Analysis")
    st.write("Track daily revenue and volume to identify seasonal peaks and overall growth.")

    try:
        # Load the trend data
        df_trend = pd.read_csv("m6_sales_trends.csv")
        
        # Convert date back to datetime for Streamlit plotting
        df_trend['Sale Date'] = pd.to_datetime(df_trend['Sale Date'])
        
        # Create a toggle so the user can switch between looking at Revenue vs Quantity
        metric_choice = st.radio("Select Metric to Analyze:", ["Total Sales (Revenue)", "Total Quantity (Volume)"])
        
        # Set the Y-axis based on the user's choice
        y_axis = 'Total_Sales' if metric_choice == "Total Sales (Revenue)" else 'Total_Quantity'
        
        # Display the interactive Streamlit Line Chart
        st.line_chart(
            data=df_trend, 
            x='Sale Date', 
            y=y_axis,
            use_container_width=True
        )
        
        # Show the raw data below the chart in a collapsible expander
        with st.expander("View Raw Trend Data"):
            st.dataframe(df_trend, use_container_width=True)
            
    except FileNotFoundError:
        st.warning("⚠️ Module 6 data not found. Please run your Jupyter code to generate 'm6_sales_trends.csv'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
