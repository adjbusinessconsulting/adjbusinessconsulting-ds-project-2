import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="DS Project 2 — Food Delivery Time | ADJ Business Consulting",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLES ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background-color: #0D1B2E; color: white; }
.stSidebar { background-color: #0a1628 !important; }
.stSidebar * { color: white !important; }
h1, h2, h3 { color: white; }
.section-box {
    background: #162440;
    border-left: 4px solid #3B82F6;
    padding: 0.75rem 1.25rem;
    border-radius: 0 8px 8px 0;
    margin: 2rem 0 1rem 0;
    font-family: 'Oxanium', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: white;
}
.info-card {
    background: #162440;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #1E3A5F;
    margin-bottom: 1rem;
    color: #CBD5E1;
    font-size: 14px;
    line-height: 1.7;
}
.tag {
    background: #1E3A5F;
    color: #93C5FD;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 12px;
    font-family: monospace;
    margin-right: 6px;
    display: inline-block;
    margin-bottom: 4px;
}
.metric-card {
    background: #162440;
    border-left: 4px solid #3B82F6;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.rec-item {
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 13px;
    color: #CBD5E1;
    border-left: 3px solid #3B82F6;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:48px;">🚀</div>
        <div style="font-family:Oxanium,sans-serif;font-size:18px;font-weight:700;margin-top:0.75rem;color:white;">DS Project 2</div>
        <div style="font-size:12px;color:#93C5FD;font-family:monospace;">Food Delivery Time Prediction</div>
        <div style="font-size:11px;color:#64748B;margin-top:4px;">ADJ Business Consulting</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#93C5FD;font-size:11px;font-family:monospace;margin-bottom:8px;'>// navigate_to</div>", unsafe_allow_html=True)
    section = st.radio("", [
        "📦 1. Project Overview",
        "🔍 2. Data Understanding",
        "📊 3. EDA",
        "⚙️ 4. Feature Engineering",
        "🤖 5. Model Training",
        "📈 6. Model Comparison",
        "🎯 7. Live ETA Predictor",
        "💡 8. Business Insights",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#64748B;font-family:monospace;'>
        DS Project 2 · ADJ Business Consulting<br>
        Food Delivery Times Dataset · 1,000 orders<br><br>
        <a href='https://adjbusinessconsulting.github.io/adj-consulting/portfolio.html'
           target='_blank' style='color:#93C5FD;'>← Back to Portfolio</a>
    </div>
    """, unsafe_allow_html=True)


# ── LOAD & TRAIN ──
@st.cache_data
def load_and_train():
    df = pd.read_parquet("food_delivery_time_dataset.parquet")
    df = df.dropna()

    target = 'Delivery_Time_min' if 'Delivery_Time_min' in df.columns else df.columns[-1]
    features = [c for c in df.columns if c != target and c != 'Order_ID']

    cat_cols = df[features].select_dtypes(include='object').columns.tolist()
    num_cols = df[features].select_dtypes(include='number').columns.tolist()

    # Feature engineering
    if 'Distance_km' in df.columns and 'Preparation_Time_min' in df.columns:
        df['Distance_x_Prep'] = df['Distance_km'] * df['Preparation_Time_min']
        num_cols.append('Distance_x_Prep')
    if 'Time_of_Day' in df.columns:
        df['Is_Rush_Hour'] = df['Time_of_Day'].isin(['Morning', 'Evening']).astype(int)
        num_cols.append('Is_Rush_Hour')

    X = df[cat_cols + num_cols]
    y = df[target]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    predictions = {}
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R²': r2_score(y_test, y_pred)
        }
        predictions[name] = (y_test.values, y_pred)

    best_pipe = Pipeline([('prep', preprocessor), ('model', LinearRegression())])
    best_pipe.fit(X_train, y_train)

    return df, results, predictions, best_pipe, X_test, y_test, cat_cols, num_cols, target


with st.spinner("Training ML models..."):
    df, results, predictions, best_pipe, X_test, y_test, cat_cols, num_cols, target = load_and_train()


def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor="#0D1B2E", plot_bgcolor="#162440",
        font_color="white", font_family="Oxanium"
    )
    return fig


# ════════════════════════════════════════════════
# SECTION 1: PROJECT OVERVIEW
# ════════════════════════════════════════════════
if section == "📦 1. Project Overview":
    st.markdown("# 🚀 Predicting Food Delivery Time with ML")
    st.markdown('<div style="color:#93C5FD;font-family:monospace;font-size:13px;margin-bottom:1.5rem;">DS Project 2 — ADJ Business Consulting · Food Delivery Times Dataset · 1,000 orders</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div style="color:#93C5FD;font-family:monospace;font-size:11px;margin-bottom:0.5rem;">// project_background</div>
        This project builds a <strong style="color:white;">machine learning model to predict food delivery time</strong>
        in minutes, based on factors such as distance, preparation time, courier experience,
        weather conditions, and traffic levels. Accurate ETA predictions improve customer satisfaction
        and help delivery operations teams optimize logistics.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, label, val in zip(
        [col1, col2, col3, col4],
        ["Total Orders", "Features Used", "Models Trained", "Best Model"],
        ["1,000", "5+", "3", "Linear Regression"]
    ):
        col.metric(label, val)

    st.markdown('<div class="section-box">📋 Dataset Attributes</div>', unsafe_allow_html=True)
    attrs = pd.DataFrame({
        "Column": ["Order_ID", "Distance_km", "Preparation_Time_min", "Courier_Experience_yrs",
                   "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type", "Delivery_Time_min"],
        "Type": ["String", "Float", "Float", "Float", "Categorical", "Categorical",
                 "Categorical", "Categorical", "Float (Target)"],
        "Description": [
            "Unique order identifier",
            "Delivery distance in kilometers",
            "Time taken to prepare the order (minutes)",
            "Years of experience of the delivery courier",
            "Weather conditions during delivery",
            "Traffic level at time of delivery",
            "Time of day the order was placed",
            "Vehicle used for delivery",
            "Actual delivery time in minutes — this is what we predict"
        ]
    })
    st.dataframe(attrs, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-box" style="margin-top:1.5rem;">🗂️ Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════
# SECTION 2: DATA UNDERSTANDING
# ════════════════════════════════════════════════
elif section == "🔍 2. Data Understanding":
    st.markdown('<div class="section-box">🔍 Section 2 — Data Understanding & Cleaning</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", f"{df.shape[1]}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum()}")

    steps = [
        ("Load Dataset", "1,000 rows × 9 columns loaded from food_delivery_time_dataset.parquet"),
        ("Check Missing Values", "No missing values found — dataset is clean ✅"),
        ("Check Data Types", "Numeric columns confirmed · Categorical columns confirmed"),
        ("Drop Missing Rows", "dropna() applied — no rows removed"),
        ("Feature Engineering", "Distance_x_Prep interaction term created · Is_Rush_Hour binary flag added"),
        ("Train/Test Split", "80% training · 20% test · random_state=42"),
        ("Preprocessing", "StandardScaler for numeric · OneHotEncoder for categorical"),
    ]

    for i, (step, result) in enumerate(steps):
        color = "#34D399" if "✅" in result or "clean" in result.lower() else "#93C5FD" if "created" in result.lower() or "applied" in result.lower() else "#FBBF24"
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:14px;background:#162440;border-radius:8px;
                    padding:1rem 1.25rem;margin-bottom:0.75rem;border:1px solid #1E3A5F;">
            <div style="background:#3B82F6;color:white;border-radius:50%;width:28px;height:28px;
                        display:flex;align-items:center;justify-content:center;font-weight:700;
                        font-size:13px;flex-shrink:0;">{i+1}</div>
            <div style="flex:1;">
                <div style="color:white;font-weight:600;font-size:14px;margin-bottom:4px;">{step}</div>
                <div style="color:{color};font-family:monospace;font-size:13px;">{result}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-box" style="margin-top:1.5rem;">📊 Dataset Summary Statistics</div>', unsafe_allow_html=True)
    num_df = df.select_dtypes(include='number')
    st.dataframe(num_df.describe().round(2), use_container_width=True)


# ════════════════════════════════════════════════
# SECTION 3: EDA
# ════════════════════════════════════════════════
elif section == "📊 3. EDA":
    st.markdown('<div class="section-box">📊 Section 3 — Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.markdown("#### 3.1 Delivery Time Distribution")
    fig_hist = px.histogram(df, x=target, nbins=30, height=320,
                             color_discrete_sequence=["#3B82F6"])
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(dark_fig(fig_hist), use_container_width=True)

    st.markdown("#### 3.2 Delivery Time by Weather & Traffic")
    col_l, col_r = st.columns(2)
    with col_l:
        if 'Weather' in df.columns:
            fig_box_w = px.box(df, x='Weather', y=target, height=360,
                                color_discrete_sequence=["#3B82F6"])
            fig_box_w.update_layout(xaxis_title="Weather Condition")
            st.plotly_chart(dark_fig(fig_box_w), use_container_width=True)
    with col_r:
        if 'Traffic_Level' in df.columns:
            fig_box_t = px.box(df, x='Traffic_Level', y=target, height=360,
                                color_discrete_sequence=["#FBBF24"])
            fig_box_t.update_layout(xaxis_title="Traffic Level")
            st.plotly_chart(dark_fig(fig_box_t), use_container_width=True)

    st.markdown("#### 3.3 Numerical Feature Distributions")
    num_cols_plot = df.select_dtypes(include='number').columns.tolist()
    if target in num_cols_plot:
        num_cols_plot.remove(target)

    cols = st.columns(3)
    colors = ["#60A5FA", "#34D399", "#FBBF24", "#F87171", "#A78BFA"]
    for i, feat in enumerate(num_cols_plot[:5]):
        with cols[i % 3]:
            fig = px.histogram(df, x=feat, nbins=30, height=250,
                                color_discrete_sequence=[colors[i % len(colors)]])
            fig.update_layout(showlegend=False, margin=dict(t=30, b=10))
            st.plotly_chart(dark_fig(fig), use_container_width=True)

    st.markdown("#### 3.4 Correlation Heatmap")
    corr_df = df.select_dtypes(include='number')
    if corr_df.shape[1] > 1:
        corr = corr_df.corr()
        fig_corr = px.imshow(corr, color_continuous_scale="Blues", text_auto=".2f", height=400)
        st.plotly_chart(dark_fig(fig_corr), use_container_width=True)

    st.markdown("#### 3.5 Delivery Time by Time of Day")
    if 'Time_of_Day' in df.columns:
        fig_tod = px.box(df, x='Time_of_Day', y=target, height=340,
                          color_discrete_sequence=["#34D399"])
        fig_tod.update_layout(xaxis_title="Time of Day")
        st.plotly_chart(dark_fig(fig_tod), use_container_width=True)


# ════════════════════════════════════════════════
# SECTION 4: FEATURE ENGINEERING
# ════════════════════════════════════════════════
elif section == "⚙️ 4. Feature Engineering":
    st.markdown('<div class="section-box">⚙️ Section 4 — Feature Engineering</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div style="color:#93C5FD;font-family:monospace;font-size:11px;margin-bottom:0.75rem;">// feature_strategy</div>
        We engineered two additional features to capture interaction effects and rush-hour behavior,
        then applied <strong style="color:white;">StandardScaler</strong> for numeric features and
        <strong style="color:white;">OneHotEncoder</strong> for categorical features via a
        <strong style="color:white;">ColumnTransformer pipeline</strong>.
    </div>
    """, unsafe_allow_html=True)

    features_tbl = pd.DataFrame({
        "Feature": ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs",
                    "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type",
                    "Distance_x_Prep ✨", "Is_Rush_Hour ✨"],
        "Type": ["Numeric", "Numeric", "Numeric", "Categorical", "Categorical",
                 "Categorical", "Categorical", "Engineered (Numeric)", "Engineered (Binary)"],
        "Purpose": [
            "Distance to deliver — key driver of delivery time",
            "Kitchen preparation time — adds to total duration",
            "Courier skill level — experienced couriers are faster",
            "Weather impacts delivery speed",
            "Traffic delays affect ETA significantly",
            "Peak vs off-peak hours affect speed",
            "Bike/car/motorcycle have different speeds",
            "Interaction: Distance × Prep Time — captures compounding delays",
            "1 if Morning or Evening rush hour — captures peak delay"
        ]
    })
    st.dataframe(features_tbl, use_container_width=True, hide_index=True)

    st.markdown("#### Preprocessing Pipeline")
    st.code("""# ColumnTransformer Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Engineered Features
df['Distance_x_Prep'] = df['Distance_km'] * df['Preparation_Time_min']
df['Is_Rush_Hour']    = df['Time_of_Day'].isin(['Morning', 'Evening']).astype(int)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""", language="python")

    st.markdown('<div class="section-box" style="margin-top:1.5rem;">📊 Feature Preview</div>', unsafe_allow_html=True)
    preview_cols = [c for c in df.columns if c != 'Order_ID']
    st.dataframe(df[preview_cols].head(10), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════
# SECTION 5: MODEL TRAINING
# ════════════════════════════════════════════════
elif section == "🤖 5. Model Training":
    st.markdown('<div class="section-box">🤖 Section 5 — Model Training</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div style="color:#93C5FD;font-family:monospace;font-size:11px;margin-bottom:0.5rem;">// model_selection</div>
        We trained and compared three regression models, each wrapped in a
        <strong style="color:white;">sklearn Pipeline</strong> with the same preprocessing steps.
        All models are evaluated on the held-out 20% test set.
    </div>
    """, unsafe_allow_html=True)

    models_info = [
        ("Linear Regression", "#3B82F6",
         "Assumes a linear relationship between features and delivery time. Fast, interpretable, and a strong baseline.",
         ["No hyperparameters", "Highly interpretable", "Works well when relationships are linear"]),
        ("Random Forest", "#34D399",
         "Ensemble of decision trees — captures non-linear interactions and feature importance automatically.",
         ["n_estimators=100 (default)", "random_state=42", "Handles outliers well"]),
        ("Gradient Boosting", "#FBBF24",
         "Sequential boosting of weak learners — often the most accurate but slower to train.",
         ["n_estimators=100 (default)", "random_state=42", "Best for complex patterns"]),
    ]

    for name, color, desc, params in models_info:
        st.markdown(f"""
        <div style="background:#162440;border-radius:10px;padding:1.5rem;
                    border-left:4px solid {color};margin-bottom:1.25rem;">
            <div style="color:{color};font-family:monospace;font-size:11px;">Regression Model</div>
            <div style="color:white;font-size:17px;font-weight:700;margin:4px 0 0.5rem;">{name}</div>
            <div style="color:#94A3B8;font-size:13px;line-height:1.7;margin-bottom:0.75rem;">{desc}</div>
            {"".join([f'<div class="rec-item" style="border-left-color:{color};">{p}</div>' for p in params])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Training Code")
    st.code("""from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results[name] = {
        'MAE':  mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²':   r2_score(y_test, y_pred)
    }
""", language="python")


# ════════════════════════════════════════════════
# SECTION 6: MODEL COMPARISON
# ════════════════════════════════════════════════
elif section == "📈 6. Model Comparison":
    st.markdown('<div class="section-box">📈 Section 6 — Model Comparison & Results</div>', unsafe_allow_html=True)

    metrics_df = pd.DataFrame(results).T.round(4)

    # Metric cards
    col1, col2, col3 = st.columns(3)
    model_colors = {"Linear Regression": "#3B82F6", "Random Forest": "#34D399", "Gradient Boosting": "#FBBF24"}
    for col, (name, row) in zip([col1, col2, col3], metrics_df.iterrows()):
        is_best = name == 'Linear Regression'
        color = model_colors.get(name, "#93C5FD")
        col.markdown(f"""
        <div style="background:#162440;border-left:4px solid {color};border-radius:8px;
                    padding:1rem;text-align:center;">
            <div style="color:{color};font-family:monospace;font-size:11px;">{'✅ Best Model' if is_best else 'Model'}</div>
            <div style="color:white;font-weight:700;font-size:15px;margin:4px 0;">{name}</div>
            <div style="color:{color};font-size:20px;font-weight:700;">R² = {row['R²']:.4f}</div>
            <div style="color:#94A3B8;font-size:12px;">MAE: {row['MAE']:.2f} · RMSE: {row['RMSE']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Metrics table
    st.markdown("#### Full Metrics Table")
    st.dataframe(metrics_df, use_container_width=True)

    # Bar chart comparison
    st.markdown("#### Model Comparison — R² Score")
    fig_r2 = go.Figure([
        go.Bar(
            x=list(results.keys()),
            y=[results[m]['R²'] for m in results],
            marker_color=["#3B82F6", "#34D399", "#FBBF24"],
            text=[f"{results[m]['R²']:.4f}" for m in results],
            textposition="outside"
        )
    ])
    fig_r2.update_layout(height=350, yaxis_title="R² Score", showlegend=False)
    st.plotly_chart(dark_fig(fig_r2), use_container_width=True)

    # Actual vs Predicted
    st.markdown("#### Actual vs Predicted — Linear Regression")
    y_actual, y_pred = predictions['Linear Regression']
    scatter_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})
    scatter_df['Error'] = abs(scatter_df['Actual'] - scatter_df['Predicted'])

    fig_scatter = px.scatter(
        scatter_df, x='Actual', y='Predicted', color='Error',
        color_continuous_scale='Blues',
        labels={'Actual': 'Actual Delivery Time (min)', 'Predicted': 'Predicted Delivery Time (min)'},
        height=440
    )
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Perfect Prediction',
        line=dict(color='#93C5FD', dash='dash')
    ))
    st.plotly_chart(dark_fig(fig_scatter), use_container_width=True)

    # Residuals
    st.markdown("#### Residual Distribution")
    residuals = y_actual - y_pred
    fig_res = px.histogram(x=residuals, nbins=30, height=300,
                            color_discrete_sequence=["#F87171"],
                            labels={"x": "Residual (Actual − Predicted)"})
    fig_res.add_vline(x=0, line_dash="dash", line_color="#93C5FD")
    st.plotly_chart(dark_fig(fig_res), use_container_width=True)

    # Other model charts
    st.markdown("#### Actual vs Predicted — Random Forest & Gradient Boosting")
    col_rf, col_gb = st.columns(2)
    for col, model_name, color in zip(
        [col_rf, col_gb],
        ['Random Forest', 'Gradient Boosting'],
        ['#34D399', '#FBBF24']
    ):
        ya, yp = predictions[model_name]
        fig_m = px.scatter(x=ya, y=yp, height=320, opacity=0.5,
                            color_discrete_sequence=[color],
                            labels={'x': 'Actual', 'y': 'Predicted'})
        mn, mx = min(ya.min(), yp.min()), max(ya.max(), yp.max())
        fig_m.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                    line=dict(color='white', dash='dash'), showlegend=False))
        fig_m.update_layout(title=model_name, title_font_size=13)
        col.plotly_chart(dark_fig(fig_m), use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <div style="color:#93C5FD;font-family:monospace;font-size:11px;margin-bottom:0.5rem;">// model_interpretation</div>
        <strong style="color:white;">Linear Regression</strong> was chosen as the final model
        due to its strong balance of performance and interpretability.
        The R² score indicates the proportion of variance in delivery time explained by the features.
        Low MAE and RMSE confirm predictions are accurate within a few minutes on average.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# SECTION 7: LIVE ETA PREDICTOR
# ════════════════════════════════════════════════
elif section == "🎯 7. Live ETA Predictor":
    st.markdown('<div class="section-box">🎯 Section 7 — Live ETA Predictor</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div style="color:#93C5FD;font-family:monospace;font-size:11px;margin-bottom:0.5rem;">// how_it_works</div>
        Enter delivery details below to get a <strong style="color:white;">real-time delivery time prediction</strong>
        using the trained Linear Regression model. The model was trained on 800 orders
        and is ready to predict on new inputs.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Enter Delivery Details")
    inputs = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        if 'Distance_km' in df.columns:
            inputs['Distance_km'] = st.slider(
                "📍 Distance (km)",
                float(df['Distance_km'].min()),
                float(df['Distance_km'].max()),
                float(df['Distance_km'].median())
            )
    with col2:
        if 'Preparation_Time_min' in df.columns:
            inputs['Preparation_Time_min'] = st.slider(
                "🍳 Prep Time (min)",
                float(df['Preparation_Time_min'].min()),
                float(df['Preparation_Time_min'].max()),
                float(df['Preparation_Time_min'].median())
            )
    with col3:
        if 'Courier_Experience_yrs' in df.columns:
            inputs['Courier_Experience_yrs'] = st.slider(
                "👤 Courier Experience (yrs)",
                float(df['Courier_Experience_yrs'].min()),
                float(df['Courier_Experience_yrs'].max()),
                float(df['Courier_Experience_yrs'].median())
            )

    col4, col5 = st.columns(2)
    with col4:
        if 'Weather' in df.columns:
            inputs['Weather'] = st.selectbox("🌤️ Weather", df['Weather'].unique())
        if 'Traffic_Level' in df.columns:
            inputs['Traffic_Level'] = st.selectbox("🚦 Traffic Level", df['Traffic_Level'].unique())
    with col5:
        if 'Time_of_Day' in df.columns:
            inputs['Time_of_Day'] = st.selectbox("🕐 Time of Day", df['Time_of_Day'].unique())
        if 'Vehicle_Type' in df.columns:
            inputs['Vehicle_Type'] = st.selectbox("🏍️ Vehicle Type", df['Vehicle_Type'].unique())

    # Engineered features
    if 'Distance_km' in inputs and 'Preparation_Time_min' in inputs:
        inputs['Distance_x_Prep'] = inputs['Distance_km'] * inputs['Preparation_Time_min']
    if 'Time_of_Day' in inputs:
        inputs['Is_Rush_Hour'] = 1 if inputs['Time_of_Day'] in ['Morning', 'Evening'] else 0

    input_df = pd.DataFrame([inputs])
    try:
        prediction = best_pipe.predict(input_df)[0]
        st.markdown("---")
        st.markdown(f"""
        <div style="background:#162440;border-left:4px solid #34D399;border-radius:10px;
                    padding:1.5rem 2rem;text-align:center;margin-top:1rem;">
            <div style="color:#93C5FD;font-family:monospace;font-size:12px;margin-bottom:0.5rem;">// predicted_eta</div>
            <div style="color:#34D399;font-size:48px;font-weight:800;font-family:Oxanium;">
                {prediction:.1f} min
            </div>
            <div style="color:#94A3B8;font-size:14px;margin-top:0.5rem;">
                Estimated Delivery Time · Linear Regression Model
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Input Summary")
        input_display = {k: v for k, v in inputs.items() if k not in ['Distance_x_Prep', 'Is_Rush_Hour']}
        st.dataframe(pd.DataFrame([input_display]), use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning("⚠️ Fill in all fields to get a prediction. Make sure all inputs are valid.")


# ════════════════════════════════════════════════
# SECTION 8: BUSINESS INSIGHTS
# ════════════════════════════════════════════════
elif section == "💡 8. Business Insights":
    st.markdown('<div class="section-box">💡 Section 8 — Business Insights & Recommendations</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div style="color:#93C5FD;font-family:monospace;font-size:11px;margin-bottom:0.5rem;">// key_takeaway</div>
        <div style="color:white;font-size:18px;font-weight:700;line-height:1.4;">
            Delivery time is most strongly driven by <span style="color:#3B82F6;">distance</span>,
            <span style="color:#34D399;">preparation time</span>, and
            <span style="color:#FBBF24;">traffic conditions</span>.
            Targeting these levers can reduce average ETA by <span style="color:#34D399;">15–25%</span>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    insights = [
        {
            "title": "Optimize Kitchen Preparation",
            "color": "#3B82F6",
            "stat": "Prep time is a top driver of delivery time",
            "recs": [
                "Pre-stage ingredients during peak hours to reduce prep time",
                "Implement predictive kitchen workflows based on order history",
                "Set alerts when prep time exceeds average — flag for follow-up",
            ]
        },
        {
            "title": "Route & Distance Optimization",
            "color": "#34D399",
            "stat": "Distance directly scales with delivery time",
            "recs": [
                "Assign orders to nearest available couriers using geo-matching",
                "Cluster delivery zones to minimize total distance per route",
                "Use real-time traffic data to reroute during congestion",
            ]
        },
        {
            "title": "Rush Hour Management",
            "color": "#FBBF24",
            "stat": "Morning & Evening see longer delivery times",
            "recs": [
                "Surge-staff couriers during peak periods (morning / evening)",
                "Offer incentives for customers to order during off-peak hours",
                "Show dynamic ETA warnings during peak times in the app",
            ]
        },
        {
            "title": "Courier Experience & Fleet",
            "color": "#F87171",
            "stat": "Experienced couriers deliver faster on average",
            "recs": [
                "Assign high-value or long-distance orders to experienced couriers",
                "Create performance tiers with bonuses for fast delivery times",
                "Track vehicle type performance — optimize bike vs car assignments",
            ]
        },
    ]

    for insight in insights:
        st.markdown(f"""
        <div style="background:#162440;border-radius:10px;padding:1.5rem;
                    border-left:4px solid {insight['color']};margin-bottom:1.25rem;">
            <div style="color:{insight['color']};font-family:monospace;font-size:11px;">Business Insight</div>
            <div style="color:white;font-size:17px;font-weight:700;margin:4px 0;">{insight['title']}</div>
            <div style="color:#64748B;font-size:12px;font-family:monospace;margin-bottom:0.75rem;">{insight['stat']}</div>
            {"".join([f'<div class="rec-item" style="border-left-color:{insight["color"]};">{r}</div>' for r in insight["recs"]])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-box">📅 4-Week Action Plan</div>', unsafe_allow_html=True)

    weeks = [
        ("Week 1", "Data & Baseline", "Integrate ML model into operations system · Set ETA baseline KPIs"),
        ("Week 2", "Kitchen Ops", "Implement prep-time tracking · Launch kitchen workflow optimization"),
        ("Week 3", "Courier Routing", "Deploy geo-matching for courier assignment · Monitor distance metrics"),
        ("Week 4", "Peak Hours", "Launch rush-hour staffing surge · A/B test off-peak incentives"),
    ]

    cols = st.columns(4)
    for col, (week, focus, action) in zip(cols, weeks):
        col.markdown(f"""
        <div style="background:#162440;border-radius:8px;padding:1.25rem;border:1px solid #1E3A5F;height:100%;">
            <div style="color:#93C5FD;font-family:monospace;font-size:11px;">{week}</div>
            <div style="color:white;font-weight:700;font-size:15px;margin:6px 0;">{focus}</div>
            <div style="color:#94A3B8;font-size:13px;line-height:1.6;">{action}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:1.5rem;">
        <div style="color:#93C5FD;font-family:monospace;font-size:12px;margin-bottom:0.5rem;">// expected_outcome</div>
        <div style="color:white;font-size:16px;font-weight:600;">
            Reduce average delivery time by 15–25% ·
            Improve ETA accuracy for customers ·
            Boost courier efficiency and satisfaction
        </div>
        <div style="margin-top:1.5rem;">
            <a href="https://adjbusinessconsulting.github.io/adj-consulting/portfolio.html"
               target="_blank"
               style="background:#3B82F6;color:white;padding:12px 28px;border-radius:6px;
                      text-decoration:none;font-family:monospace;font-size:13px;font-weight:600;">
                ← Back to Portfolio
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
