import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import re
from scipy.sparse import hstack
import os





# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AutoJudge - Problem Difficulty Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.write("✅ App started")
st.write("📂 Current directory:", os.getcwd())
st.write("📁 Models folder exists:", os.path.exists("models"))

# ============================================
# CUSTOM CSS STYLING
# ============================================

st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: #f6f7fb;
    color: #111111;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #111111 !important;
}

/* Labels */
label {
    color: #111111 !important;
    font-weight: 600;
}

/* Text input & text area */
textarea, input {
    background-color: #1f2937 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* Placeholder text */
textarea::placeholder, input::placeholder {
    color: #9ca3af !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    color: #ffffff !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Buttons */
button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Info boxes */
.stAlert {
    color: #111111 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD TRAINED MODELS
# ============================================
st.write("⏳ Loading models...")

def load_models():
    """Load all trained models and preprocessors"""
    try:
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizers = pickle.load(f)

        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/svd.pkl', 'rb') as f:
            svd = pickle.load(f)
        
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('models/classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        
        with open('models/regressor.pkl', 'rb') as f:
            regressor = pickle.load(f)
        
        return vectorizers, scaler, svd, label_encoder, classifier, regressor
    
    except FileNotFoundError as e:
        st.error(f"""
        ❌ **Model files not found!**
        
        Please make sure you have these files in the 'models' folder:
        - vectorizer.pkl
        - scaler.pkl
        - svd.pkl
        - label_encoder.pkl
        - classifier.pkl
        - regressor.pkl
        
        Missing file: {str(e)}
        """)
        st.stop()
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.exception(e)
        st.stop()

# Load all models
vectorizers, scaler, svd, label_encoder, classifier, regressor = load_models()

st.sidebar.success("✅ All models loaded successfully!")


# ============================================
# PREPROCESSING FUNCTIONS (EXACT MATCH TO COLAB)
# ============================================

def clean_text(text):
    """Clean text exactly as in Colab"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    return text

def keyword_count(text):
    """Count algorithm-related keywords"""
    keywords = [
        "graph", "tree", "dp", "dynamic programming",
        "recursion", "greedy", "dfs", "bfs", "binary"
    ]
    text = text.lower()
    return sum(text.count(k) for k in keywords)

def extract_numerical_features(full_text):
    """Extract numerical features from text"""
    word_count = len(full_text.split())
    symbol_count = sum(full_text.count(s) for s in "+-*/=%")
    kw_count = keyword_count(full_text)
    
    return np.array([[word_count, symbol_count, kw_count]])

def preprocess_input(title, description, input_desc, output_desc, url=""):
    """
    Complete preprocessing function matching your Colab notebook exactly
    
    Args:
        title: Problem title
        description: Problem description
        input_desc: Input description
        output_desc: Output description
        url: Problem URL (optional, defaults to empty string)
    
    Returns:
        final_features: Processed feature vector ready for prediction
    """
    
    # Step 1: Clean each text field
    title_clean = clean_text(title)
    description_clean = clean_text(description)
    input_desc_clean = clean_text(input_desc)
    output_desc_clean = clean_text(output_desc)
    url_clean = clean_text(url)
    
    # Step 2: Create full_text by combining all fields
    full_text = f"{title_clean} {description_clean} {input_desc_clean} {output_desc_clean} {url_clean}"
    full_text = full_text.strip()
    
    # Step 3: Extract numerical features
    numerical_features = extract_numerical_features(full_text)
    
    # Step 4: Scale numerical features
    numerical_features_scaled = scaler.transform(numerical_features)
    
    # Step 5: Apply TF-IDF to each column separately with weights
    feature_weights = {
        "title": 1.0,
        "description": 1.5,
        "input_description": 1.25,
        "output_description": 1.10
    }
    
    # Create dictionary of cleaned inputs
    inputs = {
        "title": title_clean,
        "description": description_clean,
        "input_description": input_desc_clean,
        "output_description": output_desc_clean
    }
    
    # Apply vectorizers and combine
    tfidf_matrices = []
    for col, weight in feature_weights.items():
        # Transform using the saved vectorizer for this column
        X_col = vectorizers[col].transform([inputs[col]])
        # Apply weight
        X_col = X_col * weight
        tfidf_matrices.append(X_col)
    
    # Step 6: Combine all TF-IDF features
    X_tfidf_combined = hstack(tfidf_matrices)
    
    # Step 7: Apply SVD dimensionality reduction
    X_svd = svd.transform(X_tfidf_combined)
    
    # Step 8: Combine SVD features with numerical features
    final_features = np.hstack([X_svd, numerical_features_scaled])
    
    return final_features

def get_difficulty_color_class(difficulty):
    """Returns CSS class based on difficulty level"""
    difficulty_lower = difficulty.lower()
    if 'easy' in difficulty_lower:
        return 'difficulty-easy'
    elif 'medium' in difficulty_lower:
        return 'difficulty-medium'
    else:
        return 'difficulty-hard'

def predict_difficulty(title, description, input_desc, output_desc, url=""):
    """
    Main prediction function
    
    Returns:
        predicted_class: Easy/Medium/Hard
        predicted_score: Numerical difficulty score
        probabilities: Dictionary of class probabilities
    """
    # Preprocess input
    features = preprocess_input(title, description, input_desc, output_desc, url)
    
    # Predict class (encoded)
    predicted_class_encoded = classifier.predict(features)[0]
    
    # Decode class label
    predicted_class = label_encoder.inverse_transform([predicted_class_encoded])[0]
    
    # Predict score
    predicted_score = regressor.predict(features)[0]
    
    # Get class probabilities
    probabilities_encoded = classifier.predict_proba(features)[0]
    class_labels = label_encoder.inverse_transform(classifier.classes_)
    
    # Create probability dictionary
    prob_dict = {label: prob * 100 for label, prob in zip(class_labels, probabilities_encoded)}
    
    return predicted_class, predicted_score, prob_dict

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.title("ℹ️ About AutoJudge")
    
    st.markdown("""
    **AutoJudge** uses machine learning to automatically predict the difficulty 
    of programming problems based on their textual description.
    
    ### 🚀 How it works:
    1. Enter problem details
    2. Click **Predict Difficulty**
    3. Get instant AI predictions!
    
    ### 📊 Features Used:
    - **Text Features**: TF-IDF with weighted columns
    - **Numerical Features**: Word count, symbol count, keyword count
    - **Dimensionality Reduction**: SVD (1900 components)
    - **Algorithm**: Random Forest
    """)
    
    st.markdown("---")
    
    st.subheader("🤖 Model Pipeline")
    st.info("""
    1. Text Cleaning
    2. TF-IDF Vectorization (weighted)
    3. SVD Dimensionality Reduction
    4. Numerical Feature Extraction
    5. Feature Scaling
    6. Random Forest Prediction
    """)
    
    st.markdown("---")
    
    with st.expander("💡 Example Problem"):
        st.markdown("""
        **Title:** Two Sum
        
        **Description:** Given an array of integers, find two numbers that add up to a target.
        
        **Input:** Array of integers and target sum
        
        **Output:** Indices of the two numbers
        
        *Classification: Easy*
        """)

# ============================================
# MAIN APPLICATION
# ============================================

st.title("🎯 AutoJudge: Programming Problem Difficulty Predictor")
st.markdown("*Powered by Machine Learning | Built with Streamlit*")
st.markdown("---")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Problem Details")
    
    with st.form("prediction_form"):
        # Title
        title = st.text_input(
            "Problem Title",
            placeholder="e.g., Maximum Subarray Sum",
            help="Enter the problem title"
        )
        
        # Description
        description = st.text_area(
            "Problem Description",
            height=200,
            placeholder="Enter the complete problem statement...\n\nExample: Given an array of integers, find the maximum sum of any contiguous subarray.",
            help="Main problem statement"
        )
        
        # Input Description
        input_desc = st.text_area(
            "Input Description",
            height=120,
            placeholder="Describe the input format...\n\nExample: First line contains N (array size)\nSecond line contains N space-separated integers",
            help="Input format and constraints"
        )
        
        # Output Description
        output_desc = st.text_area(
            "Output Description",
            height=120,
            placeholder="Describe the output format...\n\nExample: Print a single integer representing the maximum subarray sum",
            help="Expected output format"
        )
        
        # URL (optional)
        url = st.text_input(
            "Problem URL (Optional)",
            placeholder="https://codeforces.com/problemset/problem/...",
            help="Optional: Link to the problem"
        )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Difficulty")

with col2:
    st.subheader("ℹ️ Quick Tips")
    st.markdown("""
    <div class="info-box">
    <b>For best results:</b><br><br>
    ✓ Provide complete descriptions<br>
    ✓ Include constraints<br>
    ✓ Mention algorithms if relevant<br>
    ✓ Describe edge cases<br><br>
    
    <b>Title, description, input, and output are required!</b>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PREDICTION AND RESULTS
# ============================================



if submitted:
    # Validate required inputs
    if not title.strip() or not description.strip() or not input_desc.strip() or not output_desc.strip():
        st.error("⚠️ Please fill in at least the title, description, input description, and output description!")
    else:
        with st.spinner("🔮 Analyzing problem difficulty..."):
            try:
                # Safely call prediction function
                prediction_result = predict_difficulty(title, description, input_desc, output_desc, url)

                # Ensure proper return types
                if (
                    not prediction_result 
                    or len(prediction_result) != 3
                    or not isinstance(prediction_result[2], dict)
                ):
                    st.error("❌ Prediction returned invalid results.")
                    st.stop()

                predicted_class, predicted_score, probabilities = prediction_result

                # Validate types
                if not isinstance(predicted_class, str):
                    predicted_class = "Unknown"
                if not isinstance(predicted_score, (int, float)):
                    predicted_score = 0.0
                if not isinstance(probabilities, dict) or not probabilities:
                    probabilities = {"Easy": 0.0, "Medium": 0.0, "Hard": 0.0}

                st.success("✅ Prediction completed successfully!")

                st.markdown("---")
                st.subheader("📊 Prediction Results")

                # Three columns for main results
                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    difficulty_class = get_difficulty_color_class(predicted_class) or "unknown"
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="metric-label">Difficulty Class</div>
                        <div class="metric-value">
                            <span class="{difficulty_class}">{predicted_class}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with result_col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="metric-label">Difficulty Score</div>
                        <div class="metric-value">{predicted_score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with result_col3:
                    max_prob = max(probabilities.values(), default=0.0)
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{max_prob:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Probability distribution
                st.subheader("📈 Class Probability Distribution")
                prob_df = pd.DataFrame({
                    'Difficulty Class': probabilities.keys(),
                    'Probability (%)': [float(v) for v in probabilities.values()]
                }).sort_values('Probability (%)', ascending=False)

                st.bar_chart(prob_df.set_index('Difficulty Class'))

                with st.expander("📋 View Detailed Probabilities"):
                    for difficulty, prob in sorted(probabilities.items(), key=lambda x: float(x[1]), reverse=True):
                        safe_prob = float(prob)
                        st.write(f"**{difficulty}**: {safe_prob:.2f}%")
                        st.progress(min(max(safe_prob / 100, 0.0), 1.0))  # Ensure 0-1 range

                # Interpretation
                st.markdown("---")
                st.subheader("💡 Interpretation")
                difficulty_lower = predicted_class.lower()
                if difficulty_lower == 'easy':
                    st.info("**Easy Problem**: Suitable for beginners. Involves basic programming concepts and straightforward logic.")
                elif difficulty_lower == 'medium':
                    st.warning("**Medium Problem**: Requires intermediate skills. May involve data structures, algorithms, or multi-step problem solving.")
                elif difficulty_lower == 'hard':
                    st.error("**Hard Problem**: Advanced problem requiring deep algorithmic knowledge, optimization, or complex data structures.")
                else:
                    st.info("Difficulty interpretation not available.")

                # Download results
                st.markdown("---")
                result_data = {
                    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Title': [title],
                    'Predicted Class': [predicted_class],
                    'Predicted Score': [predicted_score],
                    **{f'{k} Probability': [v] for k, v in probabilities.items()}
                }
                result_df = pd.DataFrame(result_data)
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"autojudge_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ An unexpected error occurred during prediction.")
                st.exception(e)


# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with ❤️ using Streamlit and Scikit-learn</p>
    <p>AutoJudge © 2025 | Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)
