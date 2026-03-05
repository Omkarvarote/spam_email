import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force dark theme
st.markdown("""
<script>
    window.parent.document.documentElement.setAttribute('data-theme', 'dark');
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global - Black Theme ── */
[data-testid="stAppViewContainer"] {
    background: #000000;
    color: #ffffff;
}

[data-testid="stHeader"] {
    background: #000000;
}

[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1f1f1f;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #ffffff !important;
}

/* ── Cards ── */
.card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
}
.metric-card h2 { font-size: 2rem; margin: 0; color: #ffffff; }
.metric-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #888888; }

/* ── Result banners ── */
.spam-banner {
    background: linear-gradient(135deg, #dc2626, #991b1b);
    border-radius: 14px; padding: 24px; text-align: center;
    font-size: 1.6rem; font-weight: 800; color: white;
    box-shadow: 0 8px 32px rgba(220,38,38,0.5);
    animation: pulse 1.5s ease-in-out infinite alternate;
}
.ham-banner {
    background: linear-gradient(135deg, #059669, #047857);
    border-radius: 14px; padding: 24px; text-align: center;
    font-size: 1.6rem; font-weight: 800; color: white;
    box-shadow: 0 8px 32px rgba(5,150,105,0.4);
}
@keyframes pulse {
    from { box-shadow: 0 8px 32px rgba(220,38,38,0.5); }
    to   { box-shadow: 0 8px 60px rgba(220,38,38,0.8); }
}

/* ── Section headers ── */
.section-header {
    font-size: 1.35rem; font-weight: 700; color: #8b5cf6;
    border-left: 4px solid #7c3aed;
    padding-left: 12px; margin-bottom: 16px;
}

/* ── Badges ── */
.badge-spam { background:#dc2626; color:white; padding:4px 12px;
              border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-ham  { background:#059669; color:white; padding:4px 12px;
              border-radius:20px; font-size:0.78rem; font-weight:600; }

/* ── Textarea & inputs ── */
textarea { 
    background: #1a1a1a !important;
    border: 1px solid #7c3aed !important; 
    color: white !important;
    border-radius: 10px !important; 
}

input {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: white !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 10px 28px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(124,58,237,0.6) !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] { 
    background: #1a1a1a; 
    border-radius: 12px; 
    border-bottom: 1px solid #2a2a2a;
}
[data-baseweb="tab"] { 
    color: #888888 !important; 
    font-weight: 600; 
}
[aria-selected="true"] { 
    color: #8b5cf6 !important;
    border-bottom: 3px solid #7c3aed !important; 
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: white !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
}

/* ── Tables ── */
[data-testid="stDataFrame"] { 
    border-radius: 12px; 
    overflow: hidden; 
    background: #1a1a1a;
}
thead th { 
    background: #7c3aed !important; 
    color: white !important; 
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #1a1a1a;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #2a2a2a;
}

[data-testid="stMetricLabel"] {
    color: #888888 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* ── Text and Headers ── */
h1, h2, h3, h4, h5, h6, p, span, div {
    color: inherit;
}

.stMarkdown {
    color: #ffffff;
}

/* ── Warning/Info boxes ── */
.stAlert {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    # Try to load the new model first, fall back to old one
    try:
        with open('spam_model_new.pkl', 'rb') as f:
            return pickle.load(f), 'spam_model_new.pkl'
    except FileNotFoundError:
        with open('spam_model.pkl', 'rb') as f:
            return pickle.load(f), 'spam_model.pkl'

@st.cache_data
def load_data():
    df = pd.read_csv('mail_data.csv')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Label']      = df['Category'].map({'spam': 0, 'ham': 1})
    df['msg_length'] = df['Message'].apply(len)
    df['word_count'] = df['Message'].apply(lambda x: len(x.split()))
    return df

try:
    bundle, model_file = load_models()
    tfidf  = bundle['tfidf']
    
    # Verify TF-IDF is fitted
    if not hasattr(tfidf, 'vocabulary_') or tfidf.vocabulary_ is None:
        st.error("❌ TF-IDF vectorizer is not fitted. Please train the model properly.")
        st.info("""
        **How to fix:**
        1. Run the training script to generate a new model:
        ```bash
        python train_model.py
        ```
        2. This will create `spam_model_new.pkl` with a properly fitted TF-IDF vectorizer
        3. Refresh this page
        """)
        st.stop()
    
    models = {
        'Logistic Regression': bundle['logistic_regression'],
        'Naive Bayes':         bundle['naive_bayes'],
        'Linear Regression':   bundle['linear_regression'],
    }
    accuracies   = bundle['accuracies']
    dataset_info = bundle['dataset_info']
    df = load_data()
    model_loaded = True
    
    # Show which model file is loaded
    st.sidebar.success(f"📦 Using: {model_file}")
    
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.info("""
    **To fix this issue:**
    1. Make sure you have `mail_data.csv` in the same folder
    2. Run the training script:
    ```bash
    python train_model.py
    ```
    3. This will create `spam_model_new.pkl`
    4. Refresh this page
    """)
    model_loaded = False
    st.stop()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📧 Spam Detector")
    st.markdown("---")
    
    # Start New Analysis Button
    if st.button("🔄 Start New Analysis", use_container_width=True, type="primary"):
        # Clear session state
        if 'sample' in st.session_state:
            del st.session_state['sample']
        if 'auto_analyze' in st.session_state:
            del st.session_state['auto_analyze']
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎛️ Settings")

    selected_model = st.selectbox(
        "Choose Model",
        list(models.keys()),
        index=0
    )

    st.markdown("---")
    st.markdown("### 🏆 Model Accuracy")
    for name, acc in accuracies.items():
        color = "#10b981" if acc['test'] >= 96 else "#f59e0b"
        st.markdown(f"""
        <div style='margin-bottom:8px; padding:10px 14px;
                    background:#1a1a1a; border-radius:10px;
                    border-left:3px solid {color}; border: 1px solid #2a2a2a'>
            <b style='font-size:0.8rem; color:#ffffff'>{name}</b><br>
            <span style='color:{color}; font-size:1.1rem; font-weight:800'>{acc['test']}%</span>
            <span style='color:#888; font-size:0.75rem'> test</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + Scikit-learn")

# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 10px 0 20px'>
    <h1 style='font-size:3rem; font-weight:900;
               background:linear-gradient(135deg,#8b5cf6,#06b6d4,#10b981);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        🧠 Spam Mail Detection
    </h1>
    <p style='color:#888888; font-size:1.1rem'>
        AI-powered email classifier using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍  Predict Email",
    "📊  EDA & Insights",
    "🤖  Model Comparison"
])

# ══════════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>🔍 Email Spam Detector</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Set default value from session state if sample was selected
        default_value = st.session_state.get('sample', '')
        
        email_input = st.text_area(
            "✏️ Paste your email content below:",
            value=default_value,
            height=220,
            placeholder="e.g. Congratulations! You've won a FREE prize. Click here to claim now!!!",
            key="email_text_area"
        )

        # Quick test samples
        st.markdown("**💡 Quick Test Samples:**")
        sample_cols = st.columns(2)
        spam_samples = [
            "Congratulations! You've won a FREE iPhone. Click to claim now!!!",
            "URGENT: Your bank account will be closed. Verify immediately!",
            "Get rich quick! Earn $5000 daily from home. Limited offer!",
            "FREE ENTRY: Win a luxury holiday! Call now 0800-WINNER",
        ]
        ham_samples = [
            "Hi, can we reschedule our meeting to 3pm tomorrow?",
            "Please find attached the report for Q3 review.",
            "Happy birthday! Hope you have a wonderful day.",
            "The project deadline has been moved to next Friday.",
        ]
        with sample_cols[0]:
            st.markdown("<span style='color:#ef4444; font-size:0.85rem'>🚨 Spam Samples</span>", unsafe_allow_html=True)
            for s in spam_samples:
                if st.button(f"📌 {s[:38]}…", key=f"sp_{s[:10]}", use_container_width=True):
                    st.session_state['sample'] = s
                    st.rerun()
        with sample_cols[1]:
            st.markdown("<span style='color:#10b981; font-size:0.85rem'>✅ Ham Samples</span>", unsafe_allow_html=True)
            for h in ham_samples:
                if st.button(f"📌 {h[:38]}…", key=f"hm_{h[:10]}", use_container_width=True):
                    st.session_state['sample'] = h
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button("🚀 Analyse Email", use_container_width=True)

    with col_right:
        if predict_btn and email_input.strip():
            try:
                model_obj = models[selected_model]
                features  = tfidf.transform([email_input])

                if selected_model == 'Linear Regression':
                    raw_pred = model_obj.predict(features)[0]
                    pred     = int(raw_pred >= 0.5)
                    conf     = min(abs(raw_pred - 0.5) * 2, 1.0) * 100
                else:
                    pred = model_obj.predict(features)[0]
                    prob = model_obj.predict_proba(features)[0]
                    conf = max(prob) * 100

                is_spam = (pred == 0)

                # Result banner
                if is_spam:
                    st.markdown("""
                    <div class='spam-banner'>
                        🚨 SPAM DETECTED!
                        <div style='font-size:1rem; font-weight:400; margin-top:8px'>
                            This email looks suspicious
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='ham-banner'>
                        ✅ NOT SPAM
                        <div style='font-size:1rem; font-weight:400; margin-top:8px'>
                            This email appears legitimate
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Confidence gauge
                st.markdown("**📊 Confidence Score**")
                gauge_color = "#dc2626" if is_spam else "#059669"
                st.markdown(f"""
                <div style='background:#1a1a1a; border-radius:20px; border: 1px solid #2a2a2a;
                            height:22px; overflow:hidden; margin-bottom:6px'>
                    <div style='width:{conf:.1f}%; height:100%;
                                background:linear-gradient(90deg,{gauge_color},{gauge_color});
                                border-radius:20px; transition:width 0.5s ease;
                                display:flex; align-items:center; justify-content:center;
                                font-size:0.78rem; font-weight:700; color:white'>
                        {conf:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Email stats
                st.markdown("**📝 Email Statistics**")
                words = email_input.split()
                excl  = email_input.count('!')
                upper = sum(1 for w in words if w.isupper() and len(w) > 1)
                spam_kw = ['free','win','winner','congratulations','urgent','claim',
                           'prize','cash','money','click','limited','offer','guarantee']
                kw_found = [w for w in spam_kw if w in email_input.lower()]

                stat_data = {
                    "Metric": ["📏 Characters", "📝 Words", "❗ Exclamation Marks",
                                "🔠 ALL-CAPS Words", "🔑 Spam Keywords Found"],
                    "Value": [len(email_input), len(words), excl, upper, len(kw_found)]
                }
                st.dataframe(pd.DataFrame(stat_data), hide_index=True, use_container_width=True)

                if kw_found:
                    st.markdown(f"**Spam Keywords:** " +
                        " ".join([f"<span class='badge-spam'>{k}</span>" for k in kw_found]),
                        unsafe_allow_html=True)

                st.markdown(f"<br><span style='color:#888; font-size:0.8rem'>Model used: <b>{selected_model}</b></span>",
                            unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("""
                **Possible causes:**
                - TF-IDF vectorizer is not properly fitted
                - Model file is corrupted
                - Please retrain and save the model correctly
                """)

        elif predict_btn:
            st.warning("⚠️ Please enter an email to analyse.")
        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:40px'>
                <div style='font-size:4rem'>📬</div>
                <h3 style='color:#8b5cf6'>Ready to Analyse</h3>
                <p style='color:#888888'>Paste an email on the left and click<br>
                <b>Analyse Email</b> to detect spam.</p>
            </div>""", unsafe_allow_html=True)

    # ── Batch prediction ──
    st.markdown("---")
    st.markdown("<div class='section-header'>📂 Batch Prediction (Upload CSV)</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV with a 'Message' column", type=['csv'])
    if uploaded:
        try:
            batch_df  = pd.read_csv(uploaded)
            if 'Message' not in batch_df.columns:
                st.error("CSV must have a 'Message' column.")
            else:
                try:
                    model_obj = models[selected_model]
                    feats  = tfidf.transform(batch_df['Message'].fillna(''))
                    if selected_model == 'Linear Regression':
                        preds = (model_obj.predict(feats) >= 0.5).astype(int)
                    else:
                        preds = model_obj.predict(feats)
                    batch_df['Prediction'] = ['✅ Ham' if p == 1 else '🚨 Spam' for p in preds]
                    spam_count = (preds == 0).sum()
                    ham_count  = (preds == 1).sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Emails", len(batch_df))
                    c2.metric("🚨 Spam", spam_count)
                    c3.metric("✅ Ham", ham_count)
                    st.dataframe(batch_df[['Message', 'Prediction']], use_container_width=True)
                    csv_out = batch_df.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Results", csv_out, "predictions.csv", "text/csv")
                except Exception as pred_error:
                    st.error(f"❌ Prediction error: {str(pred_error)}")
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════
#  TAB 2 — EDA
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    # Row 1 — Distribution
    col1, col2 = st.columns(2)
    with col1:
        label_counts = df['Category'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#000000')
        ax.set_facecolor('#000000')
        colors = ['#10b981', '#dc2626']
        bars = ax.bar(label_counts.index, label_counts.values, color=colors,
                      edgecolor='white', linewidth=0.6, width=0.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 15,
                    str(int(b.get_height())), ha='center', color='white', fontweight='bold')
        ax.set_title('Spam vs Ham Count', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#2a2a2a')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#000000')
        ax.set_facecolor('#000000')
        wedges, texts, autotexts = ax.pie(
            label_counts.values, labels=label_counts.index,
            autopct='%1.1f%%', colors=colors,
            startangle=90, explode=(0.04, 0.04),
            textprops={'color': 'white', 'fontsize': 12}
        )
        for at in autotexts: at.set_fontweight('bold')
        ax.set_title('Proportion', color='white', fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Row 2 — Length analysis
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#000000')
        ax.set_facecolor('#000000')
        for cat, col in zip(['ham', 'spam'], ['#10b981', '#dc2626']):
            ax.hist(df[df['Category']==cat]['msg_length'], bins=35,
                    alpha=0.7, color=col, label=cat, edgecolor='white', linewidth=0.3)
        ax.set_title('Character Length Distribution', color='white', fontweight='bold')
        ax.legend(facecolor='#0a0a0a', labelcolor='white', edgecolor='#2a2a2a')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#2a2a2a')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#000000')
        ax.set_facecolor('#000000')
        data_box = [df[df['Category']=='ham']['word_count'],
                    df[df['Category']=='spam']['word_count']]
        bp = ax.boxplot(data_box, labels=['Ham', 'Spam'], patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp['boxes'], ['#10b981', '#dc2626']):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_title('Word Count Distribution', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#2a2a2a')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Row 3 — Top words
    def get_top_words(series, n=12):
        stopwords = {'the','and','for','you','your','this','that','are','have',
                     'will','with','from','just','not','but','can','our','all',
                     'has','was','its','been','also','more','they','their','get','got'}
        all_words = ' '.join(series).lower()
        words = re.findall(r'\b[a-z]{3,}\b', all_words)
        words = [w for w in words if w not in stopwords]
        return Counter(words).most_common(n)

    col5, col6 = st.columns(2)
    with col5:
        spam_words = get_top_words(df[df['Category']=='spam']['Message'])
        words, counts = zip(*spam_words)
        fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='#000000')
        ax.set_facecolor('#000000')
        bars = ax.barh(words[::-1], counts[::-1], color='#dc2626',
                       edgecolor='white', linewidth=0.4, alpha=0.9)
        ax.set_title('Top Words in Spam', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#2a2a2a')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col6:
        ham_words = get_top_words(df[df['Category']=='ham']['Message'])
        words, counts = zip(*ham_words)
        fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='#000000')
        ax.set_facecolor('#000000')
        ax.barh(words[::-1], counts[::-1], color='#10b981',
                edgecolor='white', linewidth=0.4, alpha=0.9)
        ax.set_title('Top Words in Ham', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#2a2a2a')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Row 4 — Correlation heatmap
    st.markdown("<div class='section-header'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)
    corr = df[['msg_length', 'word_count', 'Label']].corr()
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#000000')
    ax.set_facecolor('#000000')
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, square=True, ax=ax,
                annot_kws={"color": "white", "size": 12},
                cbar_kws={"orientation": "vertical"})
    ax.tick_params(colors='white')
    col_c, _ = st.columns([1, 1])
    with col_c:
        st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════
#  TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🤖 Model Performance Comparison</div>", unsafe_allow_html=True)

    # Accuracy table
    acc_df = pd.DataFrame([
        {'Model': m, 'Train Accuracy (%)': v['train'], 'Test Accuracy (%)': v['test']}
        for m, v in accuracies.items()
    ]).sort_values('Test Accuracy (%)', ascending=False).reset_index(drop=True)
    acc_df.index = acc_df.index + 1
    st.dataframe(acc_df, use_container_width=True)

    col1, col2 = st.columns(2)

    # Bar comparison
    with col1:
        model_names = acc_df['Model'].tolist()
        train_vals  = acc_df['Train Accuracy (%)'].tolist()
        test_vals   = acc_df['Test Accuracy (%)'].tolist()
        x = np.arange(len(model_names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#000000')
        ax.set_facecolor('#000000')
        b1 = ax.bar(x - w/2, train_vals, w, label='Train', color='#8b5cf6', alpha=0.9, edgecolor='white')
        b2 = ax.bar(x + w/2, test_vals,  w, label='Test',  color='#06b6d4', alpha=0.9, edgecolor='white')
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                    f'{b.get_height():.1f}%', ha='center', color='white', fontsize=8, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(model_names, color='white', fontsize=9)
        ax.set_ylim(85, 103)
        ax.set_title('Train vs Test Accuracy', color='white', fontweight='bold')
        ax.legend(facecolor='#0a0a0a', labelcolor='white', edgecolor='#2a2a2a')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#2a2a2a')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Radar chart
    with col2:
        from matplotlib.patches import FancyArrowPatch
        fig, ax = plt.subplots(figsize=(5, 4.5), subplot_kw=dict(polar=True), facecolor='#000000')
        ax.set_facecolor('#000000')
        categories = list(accuracies.keys())
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace(' ', '\n') for c in categories], color='white', size=8)

        colors_r = ['#8b5cf6', '#10b981', '#f59e0b']
        for (name, vals), c in zip(
            [('Test', [v['test'] for v in accuracies.values()]),
             ('Train',[v['train'] for v in accuracies.values()])],
            ['#06b6d4', '#8b5cf6']
        ):
            values = vals + vals[:1]
            ax.plot(angles, values, 'o-', color=c, linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.2, color=c)

        ax.set_ylim(85, 102)
        ax.set_yticklabels([]); ax.grid(color='#2a2a2a', linestyle='--', alpha=0.5)
        ax.set_title('Accuracy Radar', color='white', fontweight='bold', pad=15)
        ax.legend(loc='upper right', facecolor='#0a0a0a', labelcolor='white', fontsize=8, edgecolor='#2a2a2a')
        st.pyplot(fig, use_container_width=True)
        plt.close()
