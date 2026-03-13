# 📋 Guide pas-à-pas — Ce que VOUS devez faire

## ✅ ÉTAPE 1 — Préparer GitHub (10 min)

### 1.1 Créer un compte GitHub
- Aller sur [https://github.com](https://github.com)
- Créer un compte gratuit si vous n'en avez pas

### 1.2 Créer un nouveau dépôt
1. Cliquer sur **"+"** → **"New repository"**
2. Nom : `tp3-house-prices`
3. Visibilité : **Public** (obligatoire pour Streamlit Cloud gratuit)
4. Ne PAS cocher "Add README" (on a déjà le nôtre)
5. Cliquer **"Create repository"**

### 1.3 Installer Git sur votre machine
```bash
# Vérifier si Git est installé
git --version

# Si non installé : https://git-scm.com/downloads
```

### 1.4 Pusher le code
Ouvrir un terminal dans le dossier `app/` et exécuter :

```bash
git init
git add .
git commit -m "TP3 - House Prices ML App"
git branch -M main
git remote add origin https://github.com/VOTRE_USERNAME/tp3-house-prices.git
git push -u origin main
```

> 💡 Remplacez `VOTRE_USERNAME` par votre nom d'utilisateur GitHub.

---

## ✅ ÉTAPE 2 — Déployer sur Streamlit Cloud (5 min)

1. Aller sur [https://share.streamlit.io](https://share.streamlit.io)
2. Se connecter avec votre compte **GitHub**
3. Cliquer **"New app"**
4. Remplir :
   - **Repository** : `VOTRE_USERNAME/tp3-house-prices`
   - **Branch** : `main`
   - **Main file path** : `app.py`
5. Cliquer **"Deploy!"**

⏳ Le déploiement prend 2-3 minutes. Vous obtiendrez une URL du type :  
`https://votre-username-tp3-house-prices-app-xxxx.streamlit.app`

---

## ✅ ÉTAPE 3 — Configurer les Secrets (2 min)

> ⚠️ IMPORTANT : ne jamais mettre les mots de passe dans le code !

1. Dans Streamlit Cloud, cliquer sur les **3 points** de votre app → **"Settings"**
2. Aller dans l'onglet **"Secrets"**
3. Coller exactement ceci :

```toml
[auth]
admin_user     = "admin"
admin_password = "tp3_house2024"

[app]
secret_key = "une_chaine_aleatoire_longue_ici"
```

4. Cliquer **"Save"** — l'app redémarre automatiquement

---

## ✅ ÉTAPE 4 — Vérifier HTTPS 🔒

- Ouvrir votre URL Streamlit Cloud dans le navigateur
- Vérifier qu'il y a bien un **cadenas 🔒** dans la barre d'adresse
- L'URL doit commencer par `https://` ✅

> Streamlit Cloud gère HTTPS automatiquement, vous n'avez rien à faire.

---

## ✅ ÉTAPE 5 — Tester l'application

1. Ouvrir l'URL publique
2. Se connecter avec :
   - Utilisateur : `admin`
   - Mot de passe : `tp3_house2024`
3. Tester les 3 pages
4. Vérifier les logs dans Streamlit Cloud → **"Manage app"** → **"Logs"**

---

## 📸 Captures d'écran à faire pour le rendu

Prendre des captures de :
- [ ] Page de connexion
- [ ] Page d'accueil après connexion
- [ ] Page 1 Data — avec le dataset chargé
- [ ] Page 1 Data — onglet Visualisations (boxplot ou scatter)
- [ ] Page 2 Training — après entraînement (métriques + graphiques)
- [ ] Page 3 Prediction — après une prédiction
- [ ] L'URL publique avec le cadenas HTTPS 🔒

---

## ❓ Problèmes fréquents

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError` | Vérifiez que le module est dans `requirements.txt` |
| `FileNotFoundError: style.css` | Assurez-vous que le dossier `assets/` est bien pushé |
| `FileNotFoundError: data/...` | Poussez aussi le dossier `data/` sur GitHub |
| L'app se déconnecte | Normal — Streamlit Cloud relance l'app si inactif |
| Secrets non trouvés | Vérifiez l'onglet Secrets dans Streamlit Cloud Settings |
| Page blanche | Regarder les logs dans "Manage app" → "Logs" |
