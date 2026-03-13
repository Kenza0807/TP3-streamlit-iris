import streamlit as st
import logging
import datetime
from pathlib import Path

# ── Configuration logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()          # console / Streamlit Cloud logs
    ]
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TP3 — House Prices ML",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────

with open(Path(__file__).parent / "assets" / "style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Authentification basique ──────────────────────────────────────────────────
def check_credentials(username: str, password: str) -> bool:
    """Vérifie les identifiants via st.secrets (Streamlit Cloud) ou valeurs par défaut."""
    try:
        valid_user = st.secrets["auth"]["admin_user"]
        valid_pass = st.secrets["auth"]["admin_password"]
    except Exception:
        # Fallback si secrets.toml absent (dev local sans secrets)
        valid_user = "admin"
        valid_pass = "tp3_house2024"

    return username.strip() == valid_user and password == valid_pass


def login_form():
    """Affiche le formulaire de connexion."""
    st.title(" Connexion requise")
    st.markdown("Cette application est protégée. Veuillez vous connecter pour continuer.")

    with st.form("login_form"):
        username = st.text_input("👤 Nom d'utilisateur")
        password = st.text_input("🔑 Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter", type="primary", use_container_width=True)

    if submitted:
        # Validation des entrées
        if not username or not password:
            st.error(" Veuillez remplir tous les champs.")
            logger.warning("Tentative de connexion avec champs vides.")
            return

        if len(username) > 50 or len(password) > 100:
            st.error(" Entrées trop longues.")
            logger.warning(f"Entrée trop longue détectée — user: {len(username)} chars.")
            return

        if check_credentials(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["login_time"] = datetime.datetime.now().isoformat()
            logger.info(f"Connexion réussie — utilisateur : {username}")
            st.rerun()
        else:
            st.error(" Identifiants incorrects.")
            logger.warning(f"Échec de connexion pour l'utilisateur : {username}")


# ── Vérification de l'authentification ───────────────────────────────────────
if not st.session_state.get("authenticated", False):
    login_form()
    st.stop()

# ── App principale (accessible après connexion) ───────────────────────────────
logger.info(f"Page accueil chargée — user : {st.session_state.get('username', '?')}")

# Bouton déconnexion dans la sidebar
with st.sidebar:
    st.markdown(f" Connecté : **{st.session_state.get('username', '')}**")
    if st.button(" Se déconnecter"):
        logger.info(f"Déconnexion — user : {st.session_state.get('username', '?')}")
        for key in ["authenticated", "username", "login_time"]:
            st.session_state.pop(key, None)
        st.rerun()

st.title("🏠 TP3 — Application ML : House Prices")
st.markdown("""
Bienvenue ! Utilisez le **menu de gauche** pour naviguer entre les pages :

-  **1 Data** — Chargez et explorez votre dataset
- **2 Training** — Entraînez un modèle et visualisez ses performances
-  **3 Prediction** — Faites des prédictions interactives
""")

col1, col2, col3 = st.columns(3)
col1.info(" **1 Data**\nUpload CSV + exploration + visualisations")
col2.info(" **2 Training**\nEntraînement + métriques + graphiques")
col3.info(" **3 Prediction**\nSliders + prédiction + distribution")
