import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import urllib.parse
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
    
    if not hasattr(tfidf, 'vocabulary_') or tfidf.vocabulary_ is None:
        st.error("❌ TF-IDF vectorizer is not fitted. Please train the model properly.")
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
    st.sidebar.success(f"📦 Using: {model_file}")
    
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    model_loaded = False
    st.stop()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📧 Spam Detector")
    st.markdown("---")
    
    if st.button("🔄 Start New Analysis", use_container_width=True, type="primary"):
        for key in ['sample', 'auto_analyze', 'url_typed', 'url_auto_check']:
            st.session_state.pop(key, None)
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
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Predict Email",
    "📊  EDA & Insights",
    "🤖  Model Comparison",
    "🔗  Link Spam Checker"
])

# ══════════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>🔍 Email Spam Detector</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        default_value = st.session_state.get('sample', '')
        
        email_input = st.text_area(
            "✏️ Paste your email content below:",
            value=default_value,
            height=220,
            placeholder="e.g. Congratulations! You've won a FREE prize. Click here to claim now!!!",
            key="email_text_area"
        )

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
            for i, s in enumerate(spam_samples):
                if st.button(f"📌 {s[:38]}…", key=f"sp_{s[:10]}", use_container_width=True):
                    st.session_state['sample'] = s
                    st.session_state['auto_analyze'] = True
                    st.rerun()
                with st.expander("📋 Copy text", expanded=False):
                    st.code(s, language=None)

        with sample_cols[1]:
            st.markdown("<span style='color:#10b981; font-size:0.85rem'>✅ Ham Samples</span>", unsafe_allow_html=True)
            for i, h in enumerate(ham_samples):
                if st.button(f"📌 {h[:38]}…", key=f"hm_{h[:10]}", use_container_width=True):
                    st.session_state['sample'] = h
                    st.session_state['auto_analyze'] = True
                    st.rerun()
                with st.expander("📋 Copy text", expanded=False):
                    st.code(h, language=None)

        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn  = st.button("🚀 Analyse Email", use_container_width=True)
        auto_analyze = st.session_state.pop('auto_analyze', False)

    with col_right:
        if (predict_btn or auto_analyze) and email_input.strip():
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

        elif predict_btn or auto_analyze:
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
        ax.barh(words[::-1], counts[::-1], color='#dc2626',
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

    acc_df = pd.DataFrame([
        {'Model': m, 'Train Accuracy (%)': v['train'], 'Test Accuracy (%)': v['test']}
        for m, v in accuracies.items()
    ]).sort_values('Test Accuracy (%)', ascending=False).reset_index(drop=True)
    acc_df.index = acc_df.index + 1
    st.dataframe(acc_df, use_container_width=True)

    col1, col2 = st.columns(2)

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

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4.5), subplot_kw=dict(polar=True), facecolor='#000000')
        ax.set_facecolor('#000000')
        categories = list(accuracies.keys())
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace(' ', '\n') for c in categories], color='white', size=8)

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


# ══════════════════════════════════════════════
#  TAB 4 — LINK SPAM CHECKER
# ══════════════════════════════════════════════
def analyze_url(url: str) -> dict:
    """Heuristic-based URL spam analysis."""
    flags = []
    score = 0

    raw = url.strip()
    if not raw.startswith(('http://', 'https://')):
        raw = 'http://' + raw

    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return {"error": "Could not parse URL", "verdict": "Unknown", "score": 0, "flags": []}

    hostname = parsed.hostname or ''
    path     = parsed.path or ''
    query    = parsed.query or ''
    full_url = raw.lower()

    # 1. IP address instead of domain
    ip_pattern = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
    if ip_pattern.match(hostname):
        flags.append(("🔴 IP Address as Host", "Spam sites often use raw IPs to avoid traceability."))
        score += 30

    # 2. URL shorteners
    shorteners = ['bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd','buff.ly',
                  'rebrand.ly','cutt.ly','shorturl.at','tiny.cc','bl.ink','rb.gy',
                  'clck.ru','qlink.me','han.gl','rotf.lol','shrtco.de','2u.pw']
    if hostname in shorteners or any(s in hostname for s in shorteners):
        flags.append(("🔴 URL Shortener Detected", "Shorteners can hide the true destination of a link."))
        score += 25

    # 3. Suspicious TLDs
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click',
                       '.loan', '.win', '.download', '.party', '.review', '.science',
                       '.stream', '.racing', '.gdn', '.bid', '.trade', '.accountant']
    for tld in suspicious_tlds:
        if hostname.endswith(tld):
            flags.append(("🟠 Suspicious TLD", f"The domain extension '{tld}' is commonly used in spam/phishing."))
            score += 20
            break

    # 4. Excessive subdomains
    parts = hostname.split('.')
    if len(parts) >= 5:
        flags.append(("🟠 Excessive Subdomains", f"{len(parts)-2} subdomains detected — may be used to disguise real domain."))
        score += 15

    # 5. Spam keywords in URL
    spam_keywords = ['free','win','winner','prize','gift','bonus','click','claim',
                     'verify','confirm','login','secure','update','account','bank',
                     'urgent','lucky','congratulations','offer','cheap','pharmacy',
                     'casino','bet','porn','xxx','nude','adult','hack','crack','keygen']
    found_kw = [kw for kw in spam_keywords if kw in full_url]
    if found_kw:
        flags.append(("🟠 Spam Keywords in URL", f"Found: {', '.join(found_kw[:5])}"))
        score += min(len(found_kw) * 8, 25)

    # 6. Many hyphens in domain
    hyphen_count = hostname.count('-')
    if hyphen_count >= 3:
        flags.append(("🟡 Many Hyphens in Domain", f"{hyphen_count} hyphens — phishing domains often mimic legit ones this way."))
        score += 10

    # 7. Very long URL
    if len(raw) > 200:
        flags.append(("🟡 Very Long URL", f"URL length is {len(raw)} chars — unusually long URLs can hide malicious redirects."))
        score += 10

    # 8. HTTP (not HTTPS)
    if parsed.scheme == 'http':
        flags.append(("🟡 No HTTPS", "The link uses HTTP instead of HTTPS — connection is not encrypted."))
        score += 8

    # 9. Heavy URL encoding
    if '%' in query or raw.count('%') > 3:
        flags.append(("🟡 URL Encoding Detected", "Heavy URL encoding may be used to obfuscate malicious parameters."))
        score += 10

    # 10. Unusual port
    if parsed.port and parsed.port not in (80, 443):
        flags.append(("🟠 Unusual Port", f"Port {parsed.port} is uncommon for web traffic."))
        score += 15

    # 11. Double slash in path
    if '//' in path:
        flags.append(("🟡 Double Slash in Path", "May be used to confuse URL parsers."))
        score += 8

    # Verdict
    if score == 0:
        verdict, verdict_color = "✅ Likely Safe", "#059669"
    elif score < 20:
        verdict, verdict_color = "🟡 Low Risk", "#d97706"
    elif score < 40:
        verdict, verdict_color = "🟠 Moderate Risk", "#ea580c"
    elif score < 65:
        verdict, verdict_color = "🔴 High Risk — Likely Spam", "#dc2626"
    else:
        verdict, verdict_color = "🚨 SPAM / Phishing URL", "#991b1b"

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "score": min(score, 100),
        "flags": flags,
        "hostname": hostname,
        "scheme": parsed.scheme,
        "path": path,
        "error": None,
    }


with tab4:
    st.markdown("<div class='section-header'>🔗 Link / URL Spam Checker</div>", unsafe_allow_html=True)

    # Sample URL lists
    spam_urls = [
        "http://192.168.1.1/login/verify?account=update",
        "http://bit.ly/3freeprize-win-now",
        "http://secure-bank-update.tk/confirm-account",
        "http://free-casino-win.click/bonus?claim=now",
    ]
    safe_urls = [
        "https://www.google.com",
        "https://github.com/openai/openai-python",
        "https://docs.python.org/3/library/re.html",
        "https://streamlit.io/gallery",
    ]

    # ── Read flags set by sample buttons (before any widget renders) ──
    auto_check = st.session_state.pop('url_auto_check', False)

    # KEY FIX: When a sample button was clicked, we stored the URL in 'url_pending'.
    # Now inject it directly into the widget's session-state key BEFORE the widget
    # renders — this is the only reliable way to pre-fill a keyed text_input.
    if 'url_pending' in st.session_state:
        st.session_state['url_check_input'] = st.session_state.pop('url_pending')

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("**🌐 Enter a URL to check:**")

        url_input = st.text_input(
            label="URL Input",
            label_visibility="collapsed",
            placeholder="e.g. https://free-prize-winner.xyz/claim?token=abc123",
            key="url_check_input"   # Streamlit owns this value via session state
        )

        # Quick test URL buttons
        st.markdown("**💡 Quick Test URLs:**")
        url_sample_cols = st.columns(2)

        with url_sample_cols[0]:
            st.markdown("<span style='color:#ef4444; font-size:0.85rem'>🚨 Suspicious URLs</span>", unsafe_allow_html=True)
            for su in spam_urls:
                if st.button(f"🔗 {su[:38]}…", key=f"surl_{su[:15]}", use_container_width=True):
                    st.session_state['url_pending']    = su   # inject on next rerun
                    st.session_state['url_auto_check'] = True
                    st.rerun()

        with url_sample_cols[1]:
            st.markdown("<span style='color:#10b981; font-size:0.85rem'>✅ Safe URLs</span>", unsafe_allow_html=True)
            for su in safe_urls:
                if st.button(f"🔗 {su[:38]}", key=f"hurl_{su[:15]}", use_container_width=True):
                    st.session_state['url_pending']    = su   # inject on next rerun
                    st.session_state['url_auto_check'] = True
                    st.rerun()

        check_btn = st.button("🔍 Check URL", use_container_width=True)

    with col_r:
        if (check_btn or auto_check) and url_input.strip():
            result = analyze_url(url_input.strip())

            if result["error"]:
                st.error(f"❌ {result['error']}")
            else:
                st.markdown(f"""
                <div style='background:{result["verdict_color"]}; border-radius:14px; padding:22px;
                            text-align:center; font-size:1.5rem; font-weight:800; color:white;
                            box-shadow:0 8px 32px rgba(0,0,0,0.4); margin-bottom:16px'>
                    {result['verdict']}
                </div>""", unsafe_allow_html=True)

                st.markdown("**⚠️ Risk Score**")
                st.markdown(f"""
                <div style='background:#1a1a1a; border-radius:20px; border:1px solid #2a2a2a;
                            height:22px; overflow:hidden; margin-bottom:6px'>
                    <div style='width:{result["score"]}%; height:100%;
                                background:{result["verdict_color"]};
                                border-radius:20px; display:flex; align-items:center;
                                justify-content:center; font-size:0.78rem;
                                font-weight:700; color:white'>
                        {result["score"]}/100
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**🔎 URL Breakdown**")
                breakdown = {
                    "Metric": ["🌐 Hostname", "🔒 Protocol", "📂 Path"],
                    "Value": [
                        result["hostname"] or "—",
                        result["scheme"].upper(),
                        result["path"] if result["path"] and result["path"] != '/' else "/",
                    ]
                }
                st.dataframe(pd.DataFrame(breakdown), hide_index=True, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if result["flags"]:
                    st.markdown("**🚩 Risk Signals Detected:**")
                    for flag_title, flag_desc in result["flags"]:
                        st.markdown(f"""
                        <div style='background:#1a1a1a; border:1px solid #2a2a2a;
                                    border-radius:10px; padding:10px 14px; margin-bottom:8px'>
                            <b style='color:#ffffff'>{flag_title}</b><br>
                            <span style='color:#888888; font-size:0.82rem'>{flag_desc}</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background:#052e16; border:1px solid #059669; border-radius:10px;
                                padding:14px 18px; color:#6ee7b7'>
                        ✅ No suspicious signals detected. This URL appears clean.
                    </div>""", unsafe_allow_html=True)

        elif check_btn or auto_check:
            st.warning("⚠️ Please enter a URL to check.")
        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:40px'>
                <div style='font-size:4rem'>🔗</div>
                <h3 style='color:#8b5cf6'>Ready to Scan</h3>
                <p style='color:#888888'>Enter any URL on the left and click<br>
                <b>Check URL</b> to detect spam links.</p>
            </div>""", unsafe_allow_html=True)

    # ── Bulk URL checker ──
    st.markdown("---")
    st.markdown("<div class='section-header'>📂 Bulk URL Check (one per line)</div>", unsafe_allow_html=True)
    bulk_input = st.text_area(
        "Bulk URL Input",
        label_visibility="collapsed",
        height=130,
        placeholder="Paste multiple URLs here, one per line:\nhttps://example.com\nhttp://free-win.xyz/prize"
    )
    bulk_btn = st.button("🔍 Check All URLs", use_container_width=False)
    if bulk_btn and bulk_input.strip():
        urls = [u.strip() for u in bulk_input.strip().splitlines() if u.strip()]
        rows = []
        for u in urls:
            r = analyze_url(u)
            rows.append({
                "URL": u[:60] + ('…' if len(u) > 60 else ''),
                "Verdict": r["verdict"],
                "Risk Score": r["score"],
                "Flags": len(r["flags"]),
            })
        bulk_df = pd.DataFrame(rows)
        st.dataframe(bulk_df, use_container_width=True, hide_index=True)
        spam_cnt = sum(1 for r in rows if r["Risk Score"] >= 40)
        safe_cnt = len(rows) - spam_cnt
        c1, c2, c3 = st.columns(3)
        c1.metric("Total URLs", len(rows))
        c2.metric("🚨 High Risk", spam_cnt)
        c3.metric("✅ Low / No Risk", safe_cnt)
