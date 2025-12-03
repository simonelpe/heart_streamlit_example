# heart_ml_app.py
"""
Heart Disease – ML Demo

Piccola applicazione Streamlit che:
- carica un dataset pubblico su malattia cardiaca
- allena un modello binario (malattia sì/no)
- mostra alcune metriche di performance
- permette di fare una previsione per un singolo paziente
"""
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import plotly.express as px

# *PAGE CONFIG 
st.set_page_config(page_title="Heart Disease – ML Demo", layout="wide", page_icon="❤️")

# *VARS  -----------------------------------------------------------
# Data path
DATA_PATH = Path("data/health/heart.csv")

# Cols selezionate come feature
FEATURE_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# Etichette leggibili in italiano per l'UI
FEATURE_LABELS = {
    "age": "Età (anni)",
    "trestbps": "Pressione a riposo (mm Hg)",
    "chol": "Colesterolo (mg/dl)",
    "thalch": "Freq. cardiaca max (bpm)",
    "oldpeak": "Depressione ST (oldpeak)",
}
TARGET_LABEL = "Presenza di malattia (target)"

# *UTILS -----------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Legge il csv e crea la colonna target binaria."""
    df = pd.read_csv(DATA_PATH)

    # num: 0 = sano, 1–4 = malattia
    df["target"] = (df["num"] > 0).astype(int)

    cols = FEATURE_COLS + ["target"]
    return df[cols]


@st.cache_resource
def train_model(df: pd.DataFrame, max_depth: int, n_estimators: int, min_samples_leaf: int):
    """
    Allena un RandomForest e calcola:
    - accuracy su train e test
    - baseline (classe più frequente)
    - importanza delle feature
    """
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # baseline: predire sempre la classe più frequente
    majority_class = int(y_test.value_counts().idxmax())
    baseline_pred = [majority_class] * len(y_test)
    baseline_acc = accuracy_score(y_test, baseline_pred) # calcola acc se y_pred fosse sempre 1 = malato contro y_true

    metrics = {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "baseline_acc": baseline_acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # importanza delle feature
    fi = pd.Series(
        model.feature_importances_,
        index=[FEATURE_LABELS[c] for c in FEATURE_COLS],
    ).sort_values(ascending=True)

    return model, metrics, fi


######################################################################
# ------------------------------ ST APP ---------------------------- #
######################################################################

df = load_data()

st.title("❤️ Heart Disease – ML Demo")
st.caption(
    "Esempio didattico: modello binario (malattia sì/no) su 5 feature "
    "numeriche. Non è uno strumento medico reale."
)

# SLIDER ------------------------------------------

with st.sidebar:
    st.header("⚙️ Iperparametri modello")

    hyp1 = st.slider(
        "Profondità massima (max_depth)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )
    
    hyp2 = st.slider(
        "Numero di alberi (n_estimators)",
        min_value=10,
        max_value=500,
        value=200,
        step=10,
    )

    hyp3 = st.slider(
        "Numero minimo di campioni (min_samples_leaf)",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )


# *PANORAMICA --------------------------------------------------------

st.subheader("🔍 Panoramica del dataset")

col_a, col_b, col_c = st.columns(3)

n_patients = len(df)
positive_rate = df["target"].mean()

col_a.metric("Numero pazienti", n_patients)
col_b.metric("Con malattia (%)", f"{positive_rate:.1%}")
col_c.metric(
    "Sani vs malati",
    f"{(1 - positive_rate):.1%} sani / {positive_rate:.1%} malati",
)

with st.expander("Mostra prime righe del dataset"):
    st.dataframe(df.head())

# ------ Challenge ------ #
# Aggiungi sotto “Panoramica” un selectbox su FEATURE_COLS e due plot uno accanto all’altro:
# Un istogramma che mostri la distribuzione della variabile selezionata
# Un violin plot sani vs malati della variabile selezionata
# Una tabella con statistiche descrittive (describe()) della variabile selezionata

# Domande: 
# “La variabile ha una distribuzione normale?”
# “Ci sono outlier?”

with st.expander("Challenge 1"):
    selected_column = st.multiselect(
            "Seleziona la colonna da visualizzare",
            FEATURE_COLS,
            default=[FEATURE_COLS[0]],
        )


    fig_hist = px.histogram(
        df,
        x=selected_column,
        title="Distribuzione della variabile selezionata",
    )
    fig_hist.update_layout(xaxis_title="Pilota", yaxis_title="Vittorie")

    fig_violin = px.violin(
        df, 
        y=selected_column, 
        x="target", 
        color="target",)
    fig_violin.update_layout(xaxis_title="Pilota", yaxis_title="Vittorie")


    col_d, col_e = st.columns(2)
    col_d.plotly_chart(fig_hist, use_container_width=True)
    col_e.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("#### Statistiche descrittive e Analisi")
    col_stats, col_questions = st.columns([1, 1])

    with col_stats:
        
        desc_stats = df[selected_column].describe()
        st.dataframe(desc_stats, use_container_width=True)

# * RF PERFORMANCE ---------------------------------------------------

model, metrics, feature_importances = train_model(df, hyp1, hyp2, hyp3)

st.subheader("📏 Performance del modello")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy su train", f"{metrics['acc_train']:.2%}")
col2.metric("Accuracy su test", f"{metrics['acc_test']:.2%}")
col3.metric(
    "Baseline (classe più frequente)",
    f"{metrics['baseline_acc']:.2%}",
)

st.caption(
    "Se l'accuracy su test è simile a quella su train e migliore della "
    "baseline, il modello sta generalizzando in modo ragionevole."
)

# Messaggio extra su possibile overfitting
gap = metrics["acc_train"] - metrics["acc_test"]
if gap > 0.2:
    st.warning(
        "Possibile overfitting: il modello va molto meglio su train "
        f"({metrics['acc_train']:.0%}) che su test "
        f"({metrics['acc_test']:.0%}). Prova ad abbassare la profondità "
        "degli alberi o ad aumentare il minimo di pazienti per foglia."
    )
elif gap > 0.1:
    st.info(
        "Leggero overfitting: il modello è più preciso su train "
        f"({metrics['acc_train']:.0%}) che su test "
        f"({metrics['acc_test']:.0%}), ma il gap è ancora accettabile."
    )
else:
    st.success(
        "Train e test hanno performance simili: il modello sembra "
        "generalizzare bene."
    )

# *CORR & FEATURE IMPORTANCE -----------------------------------------

st.subheader("📈 Correlazioni e importanza delle variabili")

col_corr, col_imp = st.columns(2)

with col_corr:
    st.markdown("**Correlazione tra variabili e target**")

    # Rename cols for corr matrix plot
    df_corr = df.copy()
    rename_map = {col: FEATURE_LABELS[col] for col in FEATURE_COLS}
    rename_map["target"] = TARGET_LABEL
    df_corr = df_corr.rename(columns=rename_map)

    # compute corr matrix
    corr = df_corr.corr()

    # corr matrix heatmap with sns
    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax_corr,
    )
    ax_corr.set_title("Matrice di correlazione")
    st.pyplot(fig_corr)

with col_imp:
    st.markdown("**Importanza delle variabili (RandomForest)**")

    # Plot feature importances barchart with matplotlib
    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    feature_importances.plot(kind="barh", ax=ax_imp)
    ax_imp.set_xlabel("Importanza (Gini)")
    ax_imp.set_ylabel("Variabile")
    ax_imp.set_title("Importanza delle variabili")
    plt.tight_layout()
    st.pyplot(fig_imp)

# --- BOXPLOT AGGIUNTIVO ---

# 1. Identifico la variabile più importante (label leggibile)
most_imp_feature_label = feature_importances.idxmax()
st.caption(f'La variabile più importante è: **{most_imp_feature_label}**')

# 2. Recupero il nome tecnico della colonna (es: "oldpeak") 
#    Invertendo il dizionario: {'Età...': 'age', ...}
label_to_col = {v: k for k, v in FEATURE_LABELS.items()}
imp_col_name = label_to_col[most_imp_feature_label]

st.markdown("---")
st.markdown(f"##### 📦 Focus: {most_imp_feature_label} vs Malattia")

# 3. Creo il boxplot con Matplotlib/Seaborn
fig_box, ax_box = plt.subplots(figsize=(8, 4))

sns.boxplot(
    data=df,
    x="target",
    y=imp_col_name,
    hue="target",     # Colora i box in base al target
    palette="Set2",
    ax=ax_box
)

# Miglioro l'estetica
ax_box.set_title(f"Distribuzione di '{most_imp_feature_label}' separata per condizione")
ax_box.set_xlabel("Condizione")
ax_box.set_ylabel(most_imp_feature_label)
# Rinomino gli assi X da 0/1 a testo
ax_box.set_xticks([0, 1])
ax_box.set_xticklabels(["Sano (0)", "Malato (1)"])

# Mostro il grafico
st.pyplot(fig_box)

# *FORM PAZIENTE ----------------------------------------------------

st.subheader("🧪 Inserisci i dati del paziente")

cols = st.columns(3)
user_input: dict[str, float] = {}

for i, col_name in enumerate(FEATURE_COLS):
    serie = df[col_name]
    min_val = float(serie.min())
    max_val = float(serie.max())
    default = float(serie.median()) # settiamo il valore di deafault sulla mediana

    label = FEATURE_LABELS[col_name]

    with cols[i % 3]: # <---- watch out
        # prenndiamo lo user input con number input di streamlit
        user_input[col_name] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default,
        )

if st.button("Predici rischio"):
    input_df = pd.DataFrame([user_input])
    proba = model.predict_proba(input_df)[0]
    pred = int(proba[1] > 0.5)

    col_res1, col_res2 = st.columns(2)
    label_risk = "ALTO" if pred == 1 else "BASSO"
    col_res1.metric("Rischio stimato", label_risk)
    col_res2.metric("Probabilità di malattia", f"{proba[1]:.1%}")

    st.write("Valori inseriti:")
    pretty_input = {FEATURE_LABELS[k]: v for k, v in user_input.items()}
    st.json(pretty_input)

    st.info(
        "⚠️ Esempio didattico su un dataset pubblico. "
        "Non è uno strumento clinico e non va usato per decisioni reali."
    )