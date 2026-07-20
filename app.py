import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# 1. PAGE INITIALIZATION: Set the page layout to wide and add a custom header icon and tab title
st.set_page_config(
    page_title="FraudShield Pro | Intelligent Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. PRESENTATION & AESTHETICS: Inject custom CSS styles for dark-mode layouts, gradient headers,
# custom neon container boxes, big metric values, and anomaly alert badges.
st.markdown("""
<style>
    /* Gradient styled main page title */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    /* Muted subtitle underneath the header */
    .subtitle {
        font-size: 1.2rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    
    /* Neon green container for clean / legitimate transactions */
    .card-legit {
        background: linear-gradient(135deg, #112d1b 0%, #08170e 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
        color: #e2e8f0;
    }
    /* Neon red container for suspected fraudulent transactions */
    .card-fraud {
        background: linear-gradient(135deg, #3b1313 0%, #1c0707 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
        color: #e2e8f0;
    }
    
    /* Metrics numbers styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        color: #cbd5e0;
        letter-spacing: 0.05em;
    }
    
    /* Anomaly text badges */
    .anomaly-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #ef4444;
        color: white;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. GEN AI ENGINE: Helper function to prompt Google Gemini. It tries the traditional SDK first
# and falls back to the newer 'google-genai' SDK if the first one fails, ensuring high reliability.
def generate_gemini_explanation(api_key, prompt):
    try:
        # Method A: Try google-generativeai client
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e1:
        try:
            # Method B: Try google-genai client
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return response.text
        except Exception as e2:
            return f"Error invoking Gemini API:\n\n* google.generativeai error: {e1}\n* google.genai error: {e2}\n\nPlease check your key and make sure you have internet access."

# 4. DATA LOADERS: Cache resource loading so we don't reload assets on every slider movement or selectbox change.
@st.cache_resource
def load_model_assets():
    """Load model, scaler, feature names, and Z-score statistics from joblib file."""
    try:
        return joblib.load('fraud_model_assets.joblib')
    except FileNotFoundError:
        return None

@st.cache_data
def load_sample_data():
    """Load sample test cases for drop-down selection."""
    try:
        return pd.read_csv('sample_test_data.csv')
    except FileNotFoundError:
        return None

# Load files
assets = load_model_assets()
sample_data = load_sample_data()

# 5. SIDEBAR: Sidebar navigation, descriptions, and Gemini API key text box
st.sidebar.markdown("## 🛡️ Control Center")
st.sidebar.info("This AI dashboard helps detect fraudulent credit card transactions and translates statistical anomalies into natural language explanations.")

# User inputs their API key here
api_key_input = st.sidebar.text_input(
    "🔑 Google Gemini API Key",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Enter your Gemini API key to enable Gen AI explanations. Alternatively, set the GEMINI_API_KEY environment variable."
)

# Use entered key or env variable key
gemini_key = api_key_input if api_key_input else os.getenv("GEMINI_API_KEY", "")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Overview")
st.sidebar.write("**Total Transactions:** 284,807")
st.sidebar.write("**Legitimate (Class 0):** 284,315 (99.83%)")
st.sidebar.write("**Fraudulent (Class 1):** 492 (0.17%)")

# 6. HEADER DESIGNS
st.markdown("<div class='main-title'>FraudShield Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Next-Generation Real-Time Credit Card Fraud Analytics & Generative Explanations</div>", unsafe_allow_html=True)

# Guard clause if model files do not exist
if assets is None:
    st.error("🚨 Trained model not found! Please run the training script first to generate model assets.")
    st.code("python train_model.py", language="bash")
    st.stop()

# Extract pre-loaded items
model = assets['model']
scaler = assets['scaler']
feature_cols = assets['feature_cols']
stats = assets['stats']
metrics_dict = assets['metrics']

# Set up Tab menus
tab_predict, tab_eda, tab_model = st.tabs([
    "🔍 Transaction Predictor", 
    "📈 Exploratory Data Analysis (EDA)", 
    "🎯 Model Performance"
])

# ----------------- TAB 1: TRANSACTION PREDICTOR -----------------
with tab_predict:
    st.header("Real-Time Transaction Predictor")
    
    # Let the user choose between choosing sample cases or typing custom values
    input_mode = st.radio(
        "Choose transaction input method:",
        ["Select Sample from Test Dataset (Recommended)", "Enter Custom Transaction Data manually"],
        horizontal=True
    )
    
    selected_row = None
    
    # Mode A: Select sample transaction
    if input_mode == "Select Sample from Test Dataset (Recommended)":
        if sample_data is None:
            st.warning("Sample test data file `sample_test_data.csv` was not found. Please train the model to generate samples.")
        else:
            # Generate descriptive titles for the dropdown list items
            labels = []
            for idx, row in sample_data.iterrows():
                actual = "Fraud" if row['Class'] == 1 else "Legitimate"
                labels.append(f"Sample #{idx+1} (Actual: {actual}, Amount: ${row['Amount']:.2f})")
                
            selected_idx = st.selectbox(
                "Pick a transaction sample from the test set:",
                range(len(labels)),
                format_func=lambda i: labels[i]
            )
            # Pick the row chosen by user
            selected_row = sample_data.iloc[selected_idx]
            
    # Mode B: Manual entry using number inputs
    else:
        st.markdown("### Manually Enter Transaction Details")
        st.info("Input transaction amount, time, and PCA components (V1-V28). Pre-filled with dataset averages.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=88.35, step=5.0)
            time = st.number_input("Transaction Time (Seconds)", min_value=0.0, value=94000.0, step=100.0)
            v_inputs = {}
            for i in range(1, 10):
                v_inputs[f'V{i}'] = st.number_input(f'V{i} (PCA component)', value=0.0, step=0.1)
        with col2:
            for i in range(10, 20):
                v_inputs[f'V{i}'] = st.number_input(f'V{i} (PCA component)', value=0.0, step=0.1)
        with col3:
            for i in range(20, 29):
                v_inputs[f'V{i}'] = st.number_input(f'V{i} (PCA component)', value=0.0, step=0.1)
                
        # Build pandas Series from manually entered fields
        custom_data = {'Time': time, 'Amount': amount}
        custom_data.update(v_inputs)
        selected_row = pd.Series(custom_data)
        
    if selected_row is not None:
        # Extract features (removing the target 'Class' column if present in the selected sample Series)
        actual_class = selected_row.get('Class', None)
        features_df = pd.DataFrame([selected_row[feature_cols]])
        
        # Scale features using the StandardScaler loaded from model assets
        features_scaled = scaler.transform(features_df)
        
        # Run ML model inference
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        fraud_prob = probabilities[1]
        
        # Map scaled features back to a dictionary for easier comparison
        transaction_scaled_dict = dict(zip(feature_cols, features_scaled[0]))
        
        # Compute deviation scores (Z-scores) comparing this transaction's V-components
        # to the legitimate class baseline. This isolates the anomalous factors.
        deviations = {}
        for col in feature_cols:
            mean_legit = stats['mean_legit'][col]
            # Add small epsilon to avoid divide-by-zero errors
            std_legit = stats['std_legit'][col]
            val_scaled = transaction_scaled_dict[col]
            z_score = (val_scaled - mean_legit) / (std_legit + 1e-6)
            deviations[col] = {
                'value_scaled': val_scaled,
                'z_score': z_score,
                'abs_z_score': abs(z_score),
                'mean_legit': mean_legit,
                'mean_fraud': stats['mean_fraud'][col]
            }
            
        # Sort features by absolute Z-score deviation in descending order
        sorted_deviations = sorted(deviations.items(), key=lambda x: x[1]['abs_z_score'], reverse=True)
        # Select the top 5 most outlying features
        top_anomalies = sorted_deviations[:5]
        
        # Render clean vs. fraud warning card designs in HTML/CSS
        st.markdown("### Decision Assessment")
        
        if prediction == 1:
            st.markdown(f"""
            <div class='card-fraud'>
                <div class='metric-label'>Detection Status</div>
                <div class='metric-value'>⚠️ HIGH FRAUD RISK ({fraud_prob:.1%})</div>
                <div>The machine learning model has flagged this transaction as highly suspicious. Urgent review is recommended.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='card-legit'>
                <div class='metric-label'>Detection Status</div>
                <div class='metric-value'>✅ CLEAN TRANSACTION ({1-fraud_prob:.1%})</div>
                <div>The machine learning model indicates this transaction is consistent with normal, legitimate spending patterns.</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Show ground truth tag if we selected a dataset sample
        if actual_class is not None:
            actual_text = "Fraud" if actual_class == 1 else "Legitimate"
            st.write(f"**Verification Info:** Actual Ground Truth: `{actual_text}` (Model Prediction: `{"Fraud" if prediction == 1 else "Legitimate"}`)")

        # Prepare description of deviations to feed into Gemini prompt
        anomalies_list = []
        for col, dev in top_anomalies:
            val = dev['value_scaled']
            z = dev['z_score']
            m_legit = dev['mean_legit']
            m_fraud = dev['mean_fraud']
            direction = "higher" if z > 0 else "lower"
            anomalies_list.append({
                'feature': col,
                'z_score': z,
                'description': f"Feature {col} is significantly {direction} than normal (Z-score: {z:.2f}). Legitimate mean is {m_legit:.2f}, while Fraud mean is {m_fraud:.2f}."
            })
            
        # Draw charts and call AI model in two columns
        col_anom, col_genai = st.columns([1, 1.2])
        
        with col_anom:
            st.markdown("### Top Feature Anomalies")
            st.write("Below are the 5 features showing the highest statistical deviation from standard legitimate transactions:")
            
            # Create interactive horizontal bar chart for top deviations using Plotly
            anom_df = pd.DataFrame({
                'Feature': [col for col, _ in top_anomalies],
                'Deviation (Z-Score)': [dev['z_score'] for _, dev in top_anomalies],
                'Magnitude': [dev['abs_z_score'] for _, dev in top_anomalies]
            })
            
            fig = px.bar(
                anom_df, 
                x='Deviation (Z-Score)', 
                y='Feature', 
                orientation='h',
                color='Deviation (Z-Score)',
                color_continuous_scale=px.colors.diverging.RdYlGn[::-1],
                title="Top 5 Outliers vs Legitimate Baseline",
                template="plotly_dark"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col_genai:
            st.markdown("### 🤖 Generative AI Analysis")
            
            # Construct natural language prompt describing transaction context and outliers
            prompt = f"""
You are an expert financial fraud investigator. You are reviewing a transaction that has been flagged as suspicious by our machine learning model with a fraud probability of {fraud_prob:.1%}.

The system has extracted the following key statistical anomalies for this transaction compared to typical legitimate (clean) transactions:
{chr(10).join(['- ' + item['description'] for item in anomalies_list])}

Transaction Amount: ${selected_row['Amount']:.2f}
Transaction Time (relative to dataset start): {selected_row['Time']} seconds

Please write a professional, natural-language explanation detailing why this transaction is highly indicative of fraud. 
Explain how these specific V-features (which represent latent behavioral dimensions of transactions, such as location, terminal type, merchant categories, etc.) deviate from clean transactions and fit the profile of known fraudulent patterns.
Structure the report as follows:
1. **Summary**: A high-level assessment of the risk.
2. **Behavioral Anomalies**: Discussing the specific feature deviations in plain business language.
3. **Investigation Action Recommendation**: Tell the analyst what steps to take (e.g. freeze card, contact customer).

Keep it concise, professional, and clear. Avoid sounding too technical about PCA, explain the deviations as security indicators instead.
"""
            
            # Call Gemini API if key is entered
            if gemini_key:
                st.write("Generative AI is analyzing the transaction anomalies...")
                with st.spinner("Calling Gemini API..."):
                    explanation = generate_gemini_explanation(gemini_key, prompt)
                st.markdown(explanation)
            else:
                # Show rule-based fallback details if key is missing
                st.warning("⚠️ Gemini API Key not provided. Enter a key in the sidebar to generate AI risk summaries.")
                st.markdown("**Local Risk Summary (Rule-Based Fallback):**")
                st.write("The transaction exhibits anomalous behavior on multiple critical dimensions. The primary drivers are:")
                for item in anomalies_list:
                    st.write(f"- **{item['feature']}**: Z-score of `{item['z_score']:.2f}` deviates significantly from the typical clean profile.")

# ----------------- TAB 2: EXPLORATORY DATA ANALYSIS (EDA) -----------------
with tab_eda:
    st.header("Exploratory Data Analysis")
    st.write("Explore key characteristics of the credit card fraud dataset.")
    
    # Load sample portion of creditcard.csv (10k rows) for rapid browser plotting
    try:
        raw_df = pd.read_csv('creditcard.csv', nrows=10000)
    except FileNotFoundError:
        raw_df = None
        
    if raw_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Amount Distribution")
            # Log-scale histogram showing amounts
            fig_amt = px.histogram(
                raw_df, 
                x="Amount", 
                color="Class",
                log_y=True,
                title="Log-Scale Transaction Amount by Class",
                color_discrete_map={0: "#10b981", 1: "#ef4444"},
                labels={"Class": "Fraud Label (0 = Clean, 1 = Fraud)"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_amt, use_container_width=True)
            
        with col2:
            st.subheader("Interaction of Key Anomaly Indicators")
            # Scatter plot of key PCA variables showing cluster separation
            fig_scatter = px.scatter(
                raw_df, 
                x="V17", 
                y="V12", 
                color="Class",
                title="V17 vs V12 Component Distribution",
                color_continuous_scale=["#10b981", "#ef4444"],
                labels={"Class": "Fraud Label"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.subheader("Feature Correlation Matrix (Top Correlated with Fraud)")
        # Calculate feature correlations against class and extract top-correlating columns
        corr = raw_df.corr()['Class'].sort_values()
        top_corr_features = list(corr.head(5).index) + list(corr.tail(6).index)
        corr_matrix = raw_df[top_corr_features].corr()
        
        # Display Plotly correlation heatmap
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Coefficient Matrix with Target Class",
            color_continuous_scale="RdBu",
            template="plotly_dark"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Raw dataset file `creditcard.csv` not found in workspace to draw global charts. Ensure it is placed in the project root.")

# ----------------- TAB 3: MODEL PERFORMANCE -----------------
with tab_model:
    st.header("Classifier Evaluation & Performance")
    st.write("Below are the evaluation metrics for the trained Random Forest model, calculated on an imbalanced test dataset (50,000 legitimate and 100 fraudulent transactions) to simulate real-world conditions.")
    
    # Extract training metrics
    cr = metrics_dict['classification_report']
    cm = metrics_dict['confusion_matrix']
    roc_auc = metrics_dict['roc_auc']
    
    # Display metrics cards
    col_acc, col_rec, col_pre, col_f1 = st.columns(4)
    with col_acc:
        st.markdown(f"""
        <div class='card-legit' style='text-align: center;'>
            <div class='metric-label'>Accuracy</div>
            <div class='metric-value'>{cr['accuracy']:.4%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_rec:
        st.markdown(f"""
        <div class='card-legit' style='text-align: center; border-color: #3b82f6;'>
            <div class='metric-label'>Recall (Fraud)</div>
            <div class='metric-value'>{cr['1']['recall']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_pre:
        st.markdown(f"""
        <div class='card-legit' style='text-align: center; border-color: #f59e0b;'>
            <div class='metric-label'>Precision (Fraud)</div>
            <div class='metric-value'>{cr['1']['precision']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_f1:
        st.markdown(f"""
        <div class='card-legit' style='text-align: center; border-color: #8b5cf6;'>
            <div class='metric-label'>ROC-AUC Score</div>
            <div class='metric-value'>{roc_auc:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_cm, col_info = st.columns(2)
    
    with col_cm:
        st.subheader("Confusion Matrix")
        # Draw confusion matrix heatmap using Plotly
        z = cm
        x = ['Predicted Legit', 'Predicted Fraud']
        y = ['Actual Legit', 'Actual Fraud']
        
        fig_cm = px.imshow(
            z, x=x, y=y,
            text_auto=True,
            title="Model Confusion Matrix",
            color_continuous_scale="Purples",
            template="plotly_dark"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_info:
        st.subheader("Modeling Approach & Engineering Details")
        st.markdown("""
        - **Model Architecture:** Scikit-Learn `RandomForestClassifier` with balanced sub-sampling.
        - **Handling Class Imbalance:** To address the severe class skew (0.17% fraud), we downsampled the majority class (legitimate transactions) in the training split to 10,000 cases, while preserving all available training fraud cases.
        - **Evaluation Protocol:** Evaluated on a testing partition with a highly realistic, severe imbalance layout (50,000 clean transactions and 100 fraudulent transactions) to prevent metric inflation.
        - **Feature Transformation:** Scaled using a `StandardScaler` to ensure time, amount, and PCA components operate on a standardized variance level.
        - **Decision Explainability:** Combines local Z-Score distance measurements from normal cases with Generative AI prompts to produce actionable security intelligence for fraud analysts.
        """)
