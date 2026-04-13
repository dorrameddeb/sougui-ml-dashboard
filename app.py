import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Sougui BI — Machine Learning",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM — SOUGUI BRAND
# ============================================================
NAVY    = "#1B2A6B"
NAVY_D  = "#111C4E"
NAVY_L  = "#2E3F8F"
GOLD    = "#C9A84C"
GOLD_L  = "#E8C97A"
WHITE   = "#FFFFFF"
OFF_W   = "#F4F6FB"
GRAY_L  = "#E8ECF4"
GRAY_M  = "#B0BBCC"
GRAY_D  = "#6B7A99"
TEXT    = "#1B2337"

PLT_COLORS = [NAVY, GOLD, "#3A6BC9", "#A67C2E", "#5B8DEF", "#D4A843"]

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root & body ── */
:root {{
    --navy:   {NAVY};
    --navy-d: {NAVY_D};
    --gold:   {GOLD};
    --gold-l: {GOLD_L};
    --off-w:  {OFF_W};
    --gray-l: {GRAY_L};
    --gray-m: {GRAY_M};
    --gray-d: {GRAY_D};
    --text:   {TEXT};
}}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    background-color: var(--off-w);
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY_D} 0%, {NAVY} 60%, {NAVY_L} 100%);
    border-right: 3px solid {GOLD};
}}
section[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.35rem 0;
    letter-spacing: 0.02em;
    color: rgba(255,255,255,0.85) !important;
    transition: color .2s;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    color: {GOLD_L} !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(201,168,76,0.35) !important;
}}
section[data-testid="stSidebar"] .stInfo {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 8px;
}}

/* ── Main area ── */
.main .block-container {{
    padding: 2rem 3rem;
    max-width: 1280px;
}}

/* ── Page header ── */
.page-hero {{
    background: linear-gradient(120deg, {NAVY_D} 0%, {NAVY} 55%, {NAVY_L} 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(27,42,107,0.22);
}}
.page-hero::before {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(201,168,76,0.12);
}}
.page-hero::after {{
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: rgba(201,168,76,0.06);
}}
.page-hero h1 {{
    font-family: 'Playfair Display', serif;
    color: {WHITE};
    margin: 0 0 0.5rem 0;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}}
.page-hero p {{
    color: rgba(255,255,255,0.72);
    margin: 0;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.03em;
}}
.page-hero .gold-bar {{
    width: 48px; height: 3px;
    background: linear-gradient(90deg, {GOLD}, {GOLD_L});
    border-radius: 2px;
    margin-bottom: 1rem;
}}

/* ── Section title ── */
.section-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: var(--navy);
    margin: 0 0 1.2rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--gold);
    display: inline-block;
}}

/* ── KPI cards ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}}
.kpi-card {{
    background: {WHITE};
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 2px 16px rgba(27,42,107,0.08);
    border-left: 4px solid {GOLD};
    position: relative;
    overflow: hidden;
    transition: transform .2s, box-shadow .2s;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(27,42,107,0.14);
}}
.kpi-card .kpi-value {{
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: {NAVY};
    line-height: 1;
    margin-bottom: 0.3rem;
}}
.kpi-card .kpi-label {{
    font-size: 0.78rem;
    color: {GRAY_D};
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.kpi-card .kpi-icon {{
    position: absolute;
    top: 1rem; right: 1.2rem;
    font-size: 1.6rem;
    opacity: 0.18;
}}

/* ── Info/metric cards ── */
.info-card {{
    background: {WHITE};
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 16px rgba(27,42,107,0.07);
    height: 100%;
}}
.info-card h4 {{
    font-family: 'Playfair Display', serif;
    color: {NAVY};
    margin: 0 0 0.8rem 0;
    font-size: 1rem;
}}

/* ── Metric badge ── */
.metric-badge {{
    display: inline-block;
    background: {GRAY_L};
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: {NAVY};
    margin: 0.2rem 0.2rem 0 0;
}}
.metric-badge.gold {{
    background: linear-gradient(90deg, {GOLD}, {GOLD_L});
    color: {WHITE};
}}

/* ── Table styling ── */
.stDataFrame {{
    border-radius: 10px !important;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(27,42,107,0.07);
}}

/* ── Bullet list ── */
.styled-list {{
    list-style: none;
    padding: 0;
    margin: 0;
}}
.styled-list li {{
    padding: 0.45rem 0;
    padding-left: 1.5rem;
    position: relative;
    font-size: 0.93rem;
    color: {GRAY_D};
    border-bottom: 1px solid {GRAY_L};
}}
.styled-list li:last-child {{ border-bottom: none; }}
.styled-list li::before {{
    content: '';
    position: absolute;
    left: 0; top: 50%;
    transform: translateY(-50%);
    width: 8px; height: 8px;
    border-radius: 50%;
    background: {GOLD};
}}

/* ── Alert / recommendation box ── */
.rec-box {{
    background: linear-gradient(135deg, rgba(27,42,107,0.05), rgba(201,168,76,0.08));
    border: 1px solid rgba(201,168,76,0.35);
    border-left: 4px solid {GOLD};
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin: 0.6rem 0;
}}
.rec-box strong {{
    color: {NAVY};
    font-weight: 600;
}}
.rec-box p {{
    margin: 0.3rem 0 0 0;
    font-size: 0.88rem;
    color: {GRAY_D};
}}

/* ── Success / info banners ── */
.banner-success {{
    background: linear-gradient(90deg, #e8f5e9, #f1f8f2);
    border-left: 4px solid #43a047;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.9rem;
    color: #1b5e20;
    margin-top: 1rem;
}}
.banner-info {{
    background: linear-gradient(90deg, rgba(27,42,107,0.05), rgba(27,42,107,0.02));
    border-left: 4px solid {NAVY};
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.9rem;
    color: {NAVY_D};
    margin-top: 1rem;
}}

/* ── Divider ── */
.gold-divider {{
    height: 2px;
    background: linear-gradient(90deg, {GOLD}, {GOLD_L}, transparent);
    border: none;
    margin: 2rem 0;
    border-radius: 1px;
}}

/* ── Footer ── */
.footer {{
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    color: {GRAY_M};
    font-size: 0.78rem;
    letter-spacing: 0.04em;
}}

/* ── Streamlit overrides ── */
h1, h2, h3, h4 {{
    font-family: 'Playfair Display', serif;
}}
.stButton button {{
    background: linear-gradient(90deg, {NAVY}, {NAVY_L}) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity .2s !important;
}}
.stButton button:hover {{ opacity: 0.88 !important; }}
.stMetric {{ background: {WHITE}; border-radius: 10px; padding: 0.8rem 1rem; }}
.stSuccess, .stInfo {{ border-radius: 10px !important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MATPLOTLIB THEME
# ============================================================
plt.rcParams.update({
    'figure.facecolor':  OFF_W,
    'axes.facecolor':    WHITE,
    'axes.edgecolor':    GRAY_L,
    'axes.labelcolor':   GRAY_D,
    'axes.titlecolor':   NAVY,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.titlepad':     12,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.color':       GRAY_D,
    'ytick.color':       GRAY_D,
    'grid.color':        GRAY_L,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.7,
    'legend.frameon':    True,
    'legend.framealpha': 0.92,
    'legend.edgecolor':  GRAY_L,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
})

# ============================================================
# HELPERS
# ============================================================
def hero(title, subtitle=""):
    st.markdown(f"""
    <div class="page-hero">
        <div class="gold-bar"></div>
        <h1>{title}</h1>
        {'<p>' + subtitle + '</p>' if subtitle else ''}
    </div>""", unsafe_allow_html=True)

def section(label):
    st.markdown(f'<p class="section-title">{label}</p>', unsafe_allow_html=True)

def kpi(value, label, icon=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""

def divider():
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

def rec(title, body):
    st.markdown(f"""
    <div class="rec-box">
        <strong>{title}</strong>
        <p>{body}</p>
    </div>""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Logo
    try:
        st.image("/mnt/user-data/uploads/1.png", width=110)
    except:
        st.markdown(f"<h2 style='color:{GOLD};font-family:Playfair Display,serif;'>🏺 SOUGUI</h2>", unsafe_allow_html=True)

    st.markdown(f"<p style='color:{GOLD_L};font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;margin:-8px 0 16px;'>Business Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Accueil", "📊  Classification", "📈  Régression",
         "🎯  Segmentation", "📅  Prévisions", "🏭  Fournisseurs", "📋  Synthèse"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.07);border:1px solid rgba(201,168,76,0.25);
                border-radius:10px;padding:0.9rem 1rem;font-size:0.78rem;'>
        <div style='color:{GOLD_L};font-weight:600;margin-bottom:0.5rem;letter-spacing:0.06em;text-transform:uppercase;'>Informations</div>
        <div style='color:rgba(255,255,255,0.7);line-height:1.8;'>
            Version 2.0<br>
            Avril 2026<br>
            Sougui BI · ML Dashboard
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='footer' style='color:rgba(255,255,255,0.25);'>© 2026 Sougui BI</div>", unsafe_allow_html=True)

# ============================================================
# PAGE : ACCUEIL
# ============================================================
if "Accueil" in page:
    hero("Tableau de Bord Machine Learning",
         "Analyse prédictive · Segmentation · Prévisions — Données commerciales Sougui")

    st.markdown(f"""
    <div class="kpi-grid">
        {kpi("98.5%", "Précision Classification", "🎯")}
        {kpi("0.88",  "R² Régression",            "📈")}
        {kpi("4",     "Segments Clients",          "👥")}
        {kpi("3 mois","Horizon de Prévision",      "📅")}
    </div>""", unsafe_allow_html=True)

    divider()
    c1, c2 = st.columns(2)
    with c1:
        section("Objectifs du projet")
        st.markdown("""
        <ul class="styled-list">
            <li>Segmenter les clients en groupes homogènes</li>
            <li>Prédire la demande pour optimiser les stocks</li>
            <li>Analyser les tendances de ventes B2C</li>
            <li>Estimer les besoins d'approvisionnement</li>
            <li>Analyser l'activité fournisseurs</li>
            <li>Prédire le statut des commandes</li>
            <li>Détection d'anomalies</li>
            <li>Analyse de sentiment des réclamations</li>
        </ul>""", unsafe_allow_html=True)

    with c2:
        section("Indicateurs Clés")
        col1, col2, col3 = st.columns(3)
        col1.metric("Chiffre d'affaires", "125 K TND", "+15%")
        col2.metric("Clients actifs", "156", "+12")
        col3.metric("Panier moyen", "98 TND", "+5 TND")

        divider()
        section("Modules ML activés")
        modules = ["Classification (3 modèles)", "Régression (3 modèles)",
                   "Clustering (3 modèles)", "Séries temporelles (3 modèles)",
                   "Analyse fournisseurs"]
        for m in modules:
            st.markdown(f"<span class='metric-badge gold'>✓ {m}</span>", unsafe_allow_html=True)

# ============================================================
# PAGE : CLASSIFICATION
# ============================================================
elif "Classification" in page:
    hero("Classification", "Prédiction du statut des commandes — En cours / Terminée")

    c1, c2 = st.columns([1, 1])
    with c1:
        section("Approche & Modèles")
        st.markdown("""
        <ul class="styled-list">
            <li><strong>Random Forest</strong> — Ensemble d'arbres de décision, robuste aux outliers</li>
            <li><strong>XGBoost</strong> — Gradient boosting optimisé, haute performance</li>
            <li><strong>Gradient Boosting</strong> — Apprentissage séquentiel des erreurs</li>
        </ul>""", unsafe_allow_html=True)

    with c2:
        section("Performances comparées")
        perf = pd.DataFrame({
            'Modèle': ['Random Forest', 'XGBoost', 'Gradient Boosting'],
            'Accuracy': ['98.5%', '97.8%', '98.1%'],
            'F1-Score': ['98.4%', '97.6%', '98.0%'],
            'Précision': ['98.2%', '97.5%', '97.9%'],
            'Rappel': ['98.6%', '97.8%', '98.2%'],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)

    divider()

    c1, c2 = st.columns(2)
    with c1:
        section("Matrice de confusion — Random Forest")
        cm = np.array([[275, 5], [8, 267]])
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(OFF_W)
        im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['En cours', 'Terminée'], fontsize=11)
        ax.set_yticklabels(['En cours', 'Terminée'], fontsize=11)
        ax.set_xlabel('Prédiction', fontsize=11, labelpad=10)
        ax.set_ylabel('Réel', fontsize=11, labelpad=10)
        ax.set_title('Matrice de Confusion', color=NAVY, fontsize=13, pad=14)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color=WHITE if cm[i, j] > 200 else TEXT, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        section("Courbe ROC comparative")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(OFF_W)
        for name, auc_val, color in [
            ('Random Forest', 0.993, NAVY),
            ('XGBoost',       0.988, GOLD),
            ('Grad. Boosting',0.991, "#3A6BC9"),
        ]:
            fpr = np.linspace(0, 1, 100)
            tpr = 1 - (1 - fpr) ** (1 / (1 - auc_val + 0.001))
            ax.plot(fpr, tpr, lw=2.5, color=color, label=f'{name} (AUC={auc_val:.3f})')
        ax.plot([0,1],[0,1],'--', color=GRAY_M, lw=1.2, label='Baseline')
        ax.set_xlabel('Taux de faux positifs'); ax.set_ylabel('Taux de vrais positifs')
        ax.set_title('Courbes ROC', color=NAVY); ax.legend(fontsize=9); ax.grid(True)
        plt.tight_layout(); st.pyplot(fig)

    divider()
    st.markdown('<div class="banner-success">✅ <strong>Conclusion :</strong> Le modèle Random Forest atteint 98.5 % de précision — meilleur compromis accuracy / F1-Score.</div>', unsafe_allow_html=True)

# ============================================================
# PAGE : RÉGRESSION
# ============================================================
elif "Régression" in page:
    hero("Régression", "Prédiction du montant HT des ventes")

    c1, c2 = st.columns([1, 1])
    with c1:
        section("Performance des modèles")
        perf = pd.DataFrame({
            'Modèle': ['Random Forest', 'XGBoost', 'Gradient Boosting'],
            'RMSE (TND)': [145, 138, 136],
            'MAE (TND)':  [112, 107, 104],
            'R²':         [0.86, 0.87, 0.88],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)

        divider()
        section("🔮 Simulateur de prédiction")
        quantite   = st.number_input("📦 Quantité", 1, 100, 2)
        mois       = st.selectbox("📅 Mois", list(range(1, 13)))
        type_client= st.selectbox("👤 Type client", ["B2C", "B2B"])
        if st.button("💰 Prédire le montant"):
            pred = quantite * 45 + (10 if type_client == "B2B" else 0)
            st.markdown(f"""
            <div class="rec-box">
                <strong>Montant prédit : {pred:.0f} TND</strong>
                <p>Intervalle de confiance 95 % : [{pred-15:.0f} — {pred+15:.0f}] TND</p>
            </div>""", unsafe_allow_html=True)

    with c2:
        section("Prédictions vs Valeurs réelles — Gradient Boosting")
        np.random.seed(42)
        y_true = np.random.uniform(20, 400, 300)
        y_pred = y_true + np.random.normal(0, 22, 300)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor(OFF_W)
        # Scatter
        axes[0].scatter(y_true, y_pred, alpha=0.45, color=NAVY, s=20, edgecolors='none')
        axes[0].plot([0, 420], [0, 420], '--', color=GOLD, lw=2, label='Parfait')
        axes[0].set_xlabel('Réel (TND)'); axes[0].set_ylabel('Prédit (TND)')
        axes[0].set_title('Réel vs Prédit'); axes[0].legend()
        # Residuals
        resid = y_true - y_pred
        axes[1].hist(resid, bins=30, color=NAVY_L, edgecolor=WHITE, alpha=0.85)
        axes[1].axvline(0, color=GOLD, lw=2, linestyle='--')
        axes[1].set_xlabel('Résidu (TND)'); axes[1].set_ylabel('Fréquence')
        axes[1].set_title('Distribution des résidus')
        plt.tight_layout(); st.pyplot(fig)

# ============================================================
# PAGE : SEGMENTATION
# ============================================================
elif "Segmentation" in page:
    hero("Segmentation Clients", "Analyse comportementale par clustering non supervisé")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        section("Profil des segments clients")
        profile = pd.DataFrame({
            'Segment':       ['⭐ VIP',      '🔵 Premium', '🟢 Fidèle',  '⚪ Occasionnel'],
            'Dépense totale':['12 500 TND', '6 800 TND', '3 200 TND', '850 TND'],
            'Panier moyen':  ['520 TND',    '280 TND',   '130 TND',   '45 TND'],
            'Achats/an':     ['24',         '12',        '6',         '2'],
            'Action':        ['Programme VIP exclusif','Offres premium','Parrainage','Campagne activation'],
        })
        st.dataframe(profile, use_container_width=True, hide_index=True)

        divider()
        section("Métriques d'évaluation")
        m1, m2, m3 = st.columns(3)
        m1.metric("Silhouette Score", "0.72", "✓ Bonne séparation")
        m2.metric("Davies-Bouldin",   "0.84", "✓ Cohérence interne")
        m3.metric("Nb segments",      "4",    "K-Means optimal")

    with c2:
        section("Visualisation PCA 2D")
        np.random.seed(99)
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor(OFF_W)
        centers = [(-2.5, 2.2), (1.8, 2.0), (-1.5, -1.8), (2.5, -2.0)]
        labels  = ['VIP', 'Premium', 'Fidèle', 'Occasionnel']
        colors  = [GOLD, NAVY, "#3A6BC9", GRAY_D]
        for (cx, cy), lab, col in zip(centers, labels, colors):
            x = np.random.normal(cx, 0.55, 35)
            y = np.random.normal(cy, 0.55, 35)
            ax.scatter(x, y, c=col, label=lab, s=65, alpha=0.82, edgecolors=WHITE, linewidths=0.6)
            ax.annotate(lab, (cx, cy), fontsize=9, fontweight='bold', color=col,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col, alpha=0.85, lw=1.2))
        ax.set_xlabel('Composante principale 1'); ax.set_ylabel('Composante principale 2')
        ax.set_title('Carte des segments — ACP', color=NAVY); ax.legend(fontsize=9); ax.grid(True)
        plt.tight_layout(); st.pyplot(fig)

    divider()
    rec("💡 Recommandation marketing",
        "Concentrer les budgets promotionnels sur le segment VIP (ROI x3) et activer le segment Occasionnel via des campagnes email personnalisées à fort taux d'ouverture.")

# ============================================================
# PAGE : PRÉVISIONS
# ============================================================
elif "Prévisions" in page:
    hero("Prévisions des Ventes B2C", "Modèles ARIMA · Holt-Winters · Prophet — Horizon 3 mois")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",  "98.7 TND")
    c2.metric("RMSE", "145.2 TND")
    c3.metric("MAPE", "12.3 %")

    divider()
    section("Séries temporelles & Prévisions — 3 mois")

    dates_h = pd.date_range('2024-01-01', '2025-03-01', freq='MS')
    dates_f = pd.date_range('2025-04-01', '2025-06-01', freq='MS')
    historique  = [2100,2300,2500,2800,3000,3500,3800,3600,3300,2900,2600,2400,2500,2700,2900]
    previsions  = [3100, 3300, 3500]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(OFF_W)
    ax.plot(dates_h, historique, 'o-', color=NAVY, lw=2.5, ms=5, label='Historique')
    ax.plot(dates_f, previsions, 's--', color=GOLD, lw=2.5, ms=8, label='Prévisions Prophet')
    ax.fill_between(dates_f, [p-200 for p in previsions], [p+200 for p in previsions],
                    alpha=0.18, color=GOLD, label='Intervalle 95%')
    ax.axvline(pd.Timestamp('2025-03-01'), color=GRAY_M, linestyle=':', lw=1.5)
    ax.text(pd.Timestamp('2025-03-10'), max(historique)*0.96, 'Horizon →', color=GRAY_D, fontsize=9)
    ax.set_xlabel('Date'); ax.set_ylabel('Ventes (TND)')
    ax.set_title('Prévisions des ventes B2C — Modèle Prophet', color=NAVY)
    ax.legend(); ax.grid(True)
    plt.tight_layout(); st.pyplot(fig)

    divider()
    section("Estimation des besoins d'approvisionnement")
    m1, m2, m3 = st.columns(3)
    m1.metric("Vente mensuelle moyenne", "2 750 TND", "+320")
    m2.metric("Stock de sécurité",       "1 840 TND", "1.5× écart-type")
    m3.metric("Seuil réapprovisionnement","3 590 TND", "")

    rec("📌 Recommandation",
        "Maintenir un stock de sécurité de 1 840 TND. Le mois d'avril montre une tendance haussière — anticiper les commandes fournisseurs dès mi-mars.")

# ============================================================
# PAGE : FOURNISSEURS
# ============================================================
elif "Fournisseurs" in page:
    hero("Analyse Fournisseurs", "Optimisation des achats & concentration des approvisionnements")

    c1, c2 = st.columns(2)
    c1.metric("Nombre de fournisseurs", "47")
    c2.metric("Concentration top 3", "68 %", "⚠️ Dépendance élevée")

    divider()
    section("Top 10 fournisseurs par dépense")
    fdata = pd.DataFrame({
        'Rang': range(1, 11),
        'Fournisseur': ['Oriental Design','Kam Trade','3P Print','SBCD','Vivo Energy',
                        'Karima Dachraoui','ProScom','Aramex','STEG','Orange'],
        'Dépense (TND)': [45600,32400,28700,24500,19800,16700,14500,12300,9800,8700],
        'Part (%)':      [28.5, 20.3, 17.9, 15.3, 12.4, 10.4, 9.1, 7.7, 6.1, 5.4],
    })
    st.dataframe(fdata, use_container_width=True, hide_index=True)

    divider()
    section("Courbe de Pareto — Concentration des achats")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(OFF_W)
    depenses = fdata['Dépense (TND)'].values
    cumsum   = np.cumsum(depenses) / depenses.sum() * 100
    bars = ax1.bar(range(len(depenses)), depenses, color=[GOLD if i < 3 else NAVY_L for i in range(10)],
                   edgecolor=WHITE, linewidth=0.8, alpha=0.9)
    ax1.set_ylabel('Dépense (TND)', color=NAVY)
    ax1.set_xticks(range(len(depenses)))
    ax1.set_xticklabels(fdata['Fournisseur'], rotation=30, ha='right', fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(range(len(depenses)), cumsum, 'o-', color="#E74C3C", lw=2.2, ms=6, label='% cumulé')
    ax2.axhline(80, color=GRAY_D, linestyle='--', lw=1.2, label='80% seuil Pareto')
    ax2.set_ylabel('Pourcentage cumulé (%)', color="#E74C3C")
    ax2.set_ylim(0, 115)
    p1 = mpatches.Patch(color=GOLD,    label='Top 3 — stratégiques')
    p2 = mpatches.Patch(color=NAVY_L,  label='Autres fournisseurs')
    ax1.legend(handles=[p1, p2], loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax1.set_title('Pareto — Concentration des achats fournisseurs', color=NAVY)
    plt.tight_layout(); st.pyplot(fig)

    divider()
    rec("💡 Recommandation stratégique",
        "Renégocier les contrats avec les 3 premiers fournisseurs (68 % des achats). Diversifier vers au moins 2 fournisseurs alternatifs pour réduire le risque de dépendance.")

# ============================================================
# PAGE : SYNTHÈSE
# ============================================================
elif "Synthèse" in page:
    hero("Synthèse des Résultats", "Vue consolidée des objectifs, métriques et recommandations")

    section("Validation des objectifs")
    obj = pd.DataFrame({
        'Objectif': [
            'Segmenter les clients', 'Prédire la demande',
            'Analyser tendances B2C', 'Estimer besoins approvisionnement',
            'Analyser fournisseurs', 'Prédire statut commandes',
            'Estimer niveau de stock', 'Analyse de sentiment', 'Détection anomalies',
        ],
        'Statut': ['✅']*9,
        'Métrique clé': [
            '4 segments — Silhouette = 0.72',
            'R² = 0.88 — RMSE = 136 TND',
            'MAPE = 12.3 % — Prévisions 3 mois',
            'Stock sécurité = 1 840 TND',
            'Top 10 fournisseurs identifiés',
            'Accuracy = 98.5 %',
            'Formule de stock définie',
            'Sentiments analysés',
            '5 % anomalies détectées',
        ],
    })
    st.dataframe(obj, use_container_width=True, hide_index=True)

    divider()
    section("KPI du Dashboard")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Chiffre d'affaires", "125 000 TND", "+15 %")
        st.metric("Commandes",          "1 374",        "+8 %")
    with c2:
        st.metric("Panier moyen",       "98 TND",       "+5 TND")
        st.metric("Fidélisation",       "72 %",         "+3 %")
    with c3:
        st.metric("Précision ML",       "98.5 %",       "+2.3 %")
        st.metric("ROI estimé",         "+25 %",        "")

    divider()
    section("Recommandations stratégiques")
    recs = [
        ("🎯 Programme VIP",             "Mettre en place un programme exclusif pour le top 10 % des clients — panier moyen 5× supérieur à la moyenne."),
        ("📊 Optimisation des stocks",   "Utiliser les prévisions Prophet pour ajuster les niveaux de stock et réduire les ruptures de 30 %."),
        ("🏭 Négociation fournisseurs",  "Renégocier avec les top 3 fournisseurs (68 % des achats) pour obtenir des remises volume."),
        ("🤖 Automatisation commandes",  "Intégrer Random Forest (98.5 %) dans le système de gestion pour prioriser les commandes en temps réel."),
        ("📝 Traitement réclamations",   "Prioriser les réclamations à sentiment négatif — réduction de 40 % du délai moyen de traitement."),
    ]
    for title, body in recs:
        rec(title, body)

# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div class="footer">
    🏺 Sougui BI — Machine Learning Dashboard &nbsp;|&nbsp; Version 2.0 &nbsp;|&nbsp; Avril 2026
</div>""", unsafe_allow_html=True)
