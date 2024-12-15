import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Restaurant Dashboard",
                   page_icon=":bar_chart:",
                   layout="wide")
st.title("Restaurant Dashboard")

# Load the CSV file
df = pd.read_csv("test.csv")
df.head()
df.info()

#Data Preparation
df['Date'] = pd.to_datetime(df['Date'])
df['Order Time'] = pd.to_datetime(df['Order Time'])
df['Serve Time'] = pd.to_datetime(df['Serve Time'])
df['Menu'] = df['Menu'].astype(str)
df['Category'] = df['Category'].astype('category')
df['Kitchen Staff'] = df['Kitchen Staff'].astype(int)
df['Drinks Staff'] = df['Drinks Staff'].astype(int)
df['Day Of Week'] = df['Day Of Week'].astype('category')
df['Month'] = df['Date'].dt.strftime('%B')
df["Time-to-Serve"] = (df["Serve Time"] - df["Order Time"]).dt.total_seconds() / 60
df['year_month'] = df['Date'].dt.to_period('M').astype(str)

# Create a new column for segmented Time-to-Serve
bins = [0, 15, 30, 45, 60, float('inf')]  # Define the range limits
labels = ['0-15', '16-30', '31-45', '46-60', '> 60']  # Corresponding labels
df["Time-to-Serve Segment"] = pd.cut(df["Time-to-Serve"], bins=bins, labels=labels, right=True)

df["Staff"] = (df["Kitchen Staff"] + df["Drinks Staff"])

#Create new dataframe
df_serve = df.groupby(['Date', 'Category', 'Hour']).agg({
    'Kitchen Staff': 'mean',
    'Drinks Staff': 'mean',
    'Time-to-Serve': 'mean'
}).reset_index()

# Fill missing Kitchen Staff and Drinks Staff values using values from the same Date and Hour but different Category
df_serve_filled = df_serve.copy()

# Iterate through the DataFrame to fill missing values
for idx, row in df_serve_filled[df_serve_filled["Kitchen Staff"].isna() | df_serve_filled["Drinks Staff"].isna()].iterrows():
    # Find the matching row in a different category
    match = df_serve_filled[
        (df_serve_filled["Date"] == row["Date"]) &
        (df_serve_filled["Hour"] == row["Hour"]) &
        (df_serve_filled["Category"] != row["Category"])
    ]
    
    if not match.empty:
        # Fill the missing values
        if pd.isna(row["Kitchen Staff"]):
            df_serve_filled.loc[idx, "Kitchen Staff"] = match["Kitchen Staff"].values[0]
        if pd.isna(row["Drinks Staff"]):
            df_serve_filled.loc[idx, "Drinks Staff"] = match["Drinks Staff"].values[0]

df_serve_filled = pd.merge(df_serve_filled, df[['Date', 'Day Of Week', 'Month']].drop_duplicates(), on='Date', how='left')

# Sidebar Filters
st.sidebar.header("Global Filters")

# Date Range Filter
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

if start_date > end_date:
    st.sidebar.error("Error: Start Date must be before or equal to End Date")

# Add a button for quick "Today" filtering
import datetime
if st.sidebar.button("Filter for Today"):
    today = datetime.date.today()
    start_date = today
    end_date = today

# Month Filter
filtered_months = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]["Month"].unique()
selected_month = st.sidebar.multiselect("Select Month", options=filtered_months, default=filtered_months)

# Category Filter
default_categories = df["Category"].unique()
selected_category = st.sidebar.multiselect("Select Category", options=default_categories, default=default_categories)

# Day of Week Filter
default_days = df["Day Of Week"].cat.categories
selected_days = st.sidebar.multiselect("Select Day(s) of Week", options=default_days, default=default_days)

# Filter Data
@st.cache_data
def filter_data(df, start_date, end_date, selected_month, selected_category, selected_days):
    return df[
        (df["Date"] >= pd.to_datetime(start_date)) &
        (df["Date"] <= pd.to_datetime(end_date)) &
        (df["Month"].isin(selected_month)) &
        (df["Category"].isin(selected_category)) &
        (df["Day Of Week"].isin(selected_days))
    ]

filtered_data = filter_data(df, start_date, end_date, selected_month, selected_category, selected_days)
filtered_serve_data = filter_data(df_serve_filled, start_date, end_date, selected_month, selected_category, selected_days)

# Calculate the summary metrics
total_menus = filtered_data["Menu"].nunique()  # Count of unique menus
total_orders = filtered_data.groupby("Date")["Order Time"].count().sum()  # Total orders grouped by day
total_sales = filtered_data["Price"].sum()  # Sum of sales

# Summary three metrics
st.markdown("### Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Menus", value=f"{total_menus}")

with col2:
    st.metric(label="Total Orders", value=f"{total_orders}")

with col3:
    st.metric(label="Total Sales", value=f"{total_sales:,.2f}")

# First Row
col1, col2 = st.columns([1, 2])

with col1:
    # Group the data by Hour and Category, then count the orders
    orders_by_hour_category = filtered_data.groupby(['Hour', 'Category']).size().unstack(fill_value=0)

    # Create a Plotly bar chart for hover functionality
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=orders_by_hour_category.index,
        y=orders_by_hour_category["food"],
        name="Food",
        marker_color='orange',
        hoverinfo='x+y+name'
    ))

    fig.add_trace(go.Bar(
        x=orders_by_hour_category.index,
        y=orders_by_hour_category["drink"],
        name="Drink",
        marker_color='skyblue',
        hoverinfo='x+y+name'
    ))

    fig.update_layout(
        title="Number of Orders by Hour",
        xaxis_title="Hour of the Day",
        yaxis_title="Number of Orders",
        barmode="stack",
        template="plotly_white",
        margin=dict(t=50, b=50)
    )

    st.plotly_chart(fig)

with col2:
    # Group by 'Hour' and calculate the average of 'Kitchen Staff' and 'Drinks Staff'
    avg_staff_by_hour = filtered_serve_data.groupby("Hour")[["Kitchen Staff", "Drinks Staff"]].mean()

    # Group by 'Hour' and calculate the average 'Time-to-Serve' for food and drinks
    avg_tts_food = filtered_serve_data[filtered_serve_data["Category"] == "food"].groupby("Hour")["Time-to-Serve"].mean()
    avg_tts_drink = filtered_serve_data[filtered_serve_data["Category"] == "drink"].groupby("Hour")["Time-to-Serve"].mean()

    # Define the bar width and offsets
    num_bars = 2  # Number of bars in each group (Kitchen Staff and Drinks Staff)
    bar_width = 0.8  # Total width for the bar group
    bar_spacing = bar_width / num_bars

    kitchen_offset = -bar_spacing / 2  # Center of the Kitchen Staff bar
    drinks_offset = bar_spacing / 2  # Center of the Drinks Staff bar

    fig = go.Figure()

    # Add bars for Kitchen and Drinks Staff
    fig.add_trace(go.Bar(
        x=avg_staff_by_hour.index,
        y=avg_staff_by_hour["Kitchen Staff"],
        name="Kitchen Staff",
        marker_color='orange',
        hoverinfo='x+y+name',
        yaxis='y1',
        offsetgroup=0  # Align bars within the group
    ))

    fig.add_trace(go.Bar(
        x=avg_staff_by_hour.index,
        y=avg_staff_by_hour["Drinks Staff"],
        name="Drinks Staff",
        marker_color='skyblue',
        hoverinfo='x+y+name',
        yaxis='y1',
        offsetgroup=1  # Align bars within the group
    ))

    # Add lines for Time-to-Serve (Food and Drink) centered over the respective bars
    fig.add_trace(go.Scatter(
        x=np.array(avg_tts_food.index) + kitchen_offset,  # Shift to center over Kitchen Staff bar
        y=avg_tts_food,
        mode='lines+markers',
        name="Time-to-Serve (Food)",
        marker_color='green',
        hoverinfo='x+y+name',
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=np.array(avg_tts_drink.index) + drinks_offset,  # Shift to center over Drinks Staff bar
        y=avg_tts_drink,
        mode='lines+markers',
        name="Time-to-Serve (Drink)",
        marker_color='purple',
        hoverinfo='x+y+name',
        yaxis='y2'
    ))

    # Update layout for dual y-axes and bar grouping
    fig.update_layout(
        title="Staff and Time-to-Serve by Hour",
        xaxis_title="Hour of the Day",
        yaxis_title="Average Staff",
        yaxis=dict(
            title="Average Staff",
            side='left'
        ),
        yaxis2=dict(
            title="Average Time-to-Serve (minutes)",
            side='right',
            overlaying='y'
        ),
        template="plotly_white",
        barmode="group",  
        bargroupgap=0.1,  
        margin=dict(t=50, b=50),
        legend_title="Category"
    )

    st.plotly_chart(fig)

# Second Row
col1, col2 = st.columns([2, 1])

with col1:
    # Group data by 'year_month' and 'Menu' to get the total orders per menu per month
    df_grouped_menu = df.groupby(['year_month', 'Menu']).size().reset_index(name='total_orders')
    df_grouped_menu['percent_change'] = (
        df_grouped_menu.groupby('Menu')['total_orders']
        .transform(lambda x: x.pct_change() * 100)
    )

    # Replace NaN values in the first month of each menu with 0
    df_grouped_menu['percent_change'] = df_grouped_menu['percent_change'].fillna(0)

    # Create heatmap data by pivoting 'year_month' and 'Menu'
    heatmap_data = df_grouped_menu.pivot(index='Menu', columns='year_month', values='percent_change')

    # Convert pivoted data back to a tidy format for Plotly
    heatmap_data_reset = heatmap_data.reset_index()
    tidy_heatmap = heatmap_data_reset.melt(id_vars='Menu', var_name='year_month', value_name='percent_change')

    # Create heatmap using Plotly graph_objects
    fig = go.Figure(data=go.Heatmap(
        z=tidy_heatmap['percent_change'],
        x=tidy_heatmap['year_month'],
        y=tidy_heatmap['Menu'],
        colorscale=[
            [0.0, "red"],    # Red for negative changes
            [0.5, "white"],  # White for 0
            [1.0, "green"]   # Green for positive changes
        ],
        zmid=0 
    ))

    fig.update_layout(
        title="Month-over-month percentage change in sales",
        xaxis_title="Year-Month",
        yaxis_title="Menu",
        template="plotly_white",
        margin=dict(t=50, b=50)
    )

    st.plotly_chart(fig)

with col2:
    top_menus = filtered_data['Menu'].value_counts().nlargest(5)
    top_menus = top_menus.sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=top_menus.values,
        y=top_menus.index,
        orientation='h',
        marker_color='#fbd568',
        hoverinfo='x+y+name'
    ))

    fig.update_layout(
        title="Top 5 Menus",
        xaxis_title="Count",
        yaxis_title="Menu",
        template="plotly_white",
        margin=dict(t=50, b=50)
    )

    st.plotly_chart(fig)

# Third Row
# Aggregate data: Number of orders and Sales
menu_count = filtered_data['Menu'].value_counts()
menu_price_sum = filtered_data.groupby('Menu')['Price'].sum()

# Sort data by count
menu_count = menu_count.sort_values(ascending=False)
menu_price_sum = menu_price_sum[menu_count.index]  # Match the same order

fig = go.Figure()

# Bar Chart for Menu Count
fig.add_trace(go.Bar(
    x=menu_count.index,
    y=menu_count,
    name="Number of Orders",
    marker_color='#fbd568',
    hoverinfo='x+y+name',  
    text=menu_count,  
    textposition='auto',  
))

# Line Chart for Sum of Price
fig.add_trace(go.Scatter(
    x=menu_count.index,
    y=menu_price_sum,
    mode='lines+markers',
    name="Sales",
    line=dict(color='mediumvioletred', width=2),
    hoverinfo='x+y+name',  
    marker=dict(color='mediumvioletred', size=8),  
    yaxis='y2',  
))

# Customize Layout
fig.update_layout(
    title='Number of Orders and Sales',
    xaxis_title='Menu',
    yaxis_title='Number of Orders',
    yaxis2=dict(
        title='Sales',
        overlaying='y',
        side='right',
        showgrid=False  
    ),
    template='plotly_white',
    hovermode='x unified'
)

st.plotly_chart(fig)
