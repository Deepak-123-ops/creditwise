# 💳 CreditWise – Loan Approval System

A Streamlit web app that predicts loan approval using a **Naïve Bayes** classifier (86% accuracy).

## 📁 Project Structure

```
creditwise/
├── app.py                  ← Streamlit application
├── train_model.py          ← Model training script
├── loan_approval_data.csv  ← Dataset
├── model.pkl               ← Trained model
├── scaler.pkl              ← StandardScaler
├── ohe.pkl                 ← OneHotEncoder
├── le_edu.pkl              ← LabelEncoder (Education)
├── feature_names.pkl       ← Column alignment
├── requirements.txt        ← Python dependencies
└── .streamlit/
    └── config.toml         ← UI theme
```

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial CreditWise deploy"
# Create a repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/creditwise.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to **https://streamlit.io/cloud** and sign in with GitHub
2. Click **"New app"**
3. Select your repo → branch `main` → main file `app.py`
4. Click **"Deploy!"**

Your app will be live at `https://YOUR_USERNAME-creditwise.streamlit.app` 🎉

## 💻 Run Locally

```bash
pip install -r requirements.txt
python train_model.py      # generates .pkl files (already done)
streamlit run app.py
```

## 📊 Model Info

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 86.0 % |
| Precision | 81.1 % |
| Recall    | 70.5 % |
| F1 Score  | 75.4 % |

Model: **Gaussian Naïve Bayes** (best precision among LR, KNN, NB)
