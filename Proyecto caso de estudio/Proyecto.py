import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# Estilo global para seaborn
sns.set_theme(style="whitegrid", font_scale=1.1)

# === Cargar datos ===
df = pd.read_csv(r'C:\Users\mikel\OneDrive\Documentos\Ventas\emails.csv')  # Cambia la ruta si es local

# --- Preprocesamiento ---
def preprocess_text(text):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['Cuerpo del mensaje'] = df['Cuerpo del mensaje'].apply(preprocess_text)

# X = textos, y = etiquetas
X = df['Cuerpo del mensaje']
y = df['Etiqueta']

# Mapear etiquetas
if y.dtype == object:
    y = y.str.strip().str.lower().map({'ham': 0, 'no spam': 0, 'nospam': 0, 'spam': 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modelo
model = make_pipeline(TfidfVectorizer(stop_words=None), MultinomialNB())
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClasificación Report:\n", classification_report(y_test, y_pred, target_names=['No Spam','Spam']))

# ===== GRÁFICAS MEJORADAS =====

# 1) Distribución Spam vs No Spam
plt.figure(figsize=(6,4))
etiquetas_legibles = df['Etiqueta'].astype(str).str.strip().str.lower().map({'ham': 'No Spam', 'spam': 'Spam'})
sns.countplot(x=etiquetas_legibles, palette="viridis")
plt.title('Distribución de Correos: Spam vs No Spam', fontsize=14, weight='bold')
plt.xlabel('Etiqueta', fontsize=12)
plt.ylabel('Número de Correos', fontsize=12)
plt.tight_layout()
plt.show()

# 2) Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu",
            xticklabels=['No Spam','Spam'], yticklabels=['No Spam','Spam'])
plt.title('Matriz de Confusión', fontsize=14, weight='bold')
plt.xlabel('Predicción', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.tight_layout()
plt.show()

# 3) Porcentaje Spam vs No Spam
porc = etiquetas_legibles.value_counts(normalize=True) * 100
plt.figure(figsize=(6,4))
sns.barplot(x=porc.index, y=porc.values, palette="mako")
plt.title('Porcentaje de Correos Spam vs No Spam', fontsize=14, weight='bold')
plt.xlabel('Etiqueta', fontsize=12)
plt.ylabel('Porcentaje (%)', fontsize=12)
for i, val in enumerate(porc.values):
    plt.text(i, val + 0.5, f"{val:.1f}%", ha='center', fontsize=11, weight='bold')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
