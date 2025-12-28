import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.set_page_config(page_title="Data Science Academy", layout="wide")


st.sidebar.title("🛠️ Proje Modülleri")
app_mode = st.sidebar.selectbox("Giriş Yapın", 
    ["Ana Sayfa", "Görüntü İşleme", "Veri Analizi (EDA)", "Boyut İndirgeme & ML"])


if app_mode == "Ana Sayfa":
    st.title("🎓 Dönem Sonu Veri Bilimi Projesi")
    st.write("Bu proje; görüntü işlemeden makine öğrenmesine kadar tüm müfredatı kapsar.")
    
    st.subheader("1. Adım: Veri Kaynağını Seçin")
    data_source = st.radio("Veriyi nereden alalım?", ["Hazır Veri Seti Kullan", "Kendi CSV Dosyamı Yükle"])
    
    if data_source == "Hazır Veri Seti Kullan":
        dataset_name = st.selectbox("Bir set seçin", ["Iris (Çiçek Türleri)", "Breast Cancer (Meme Kanseri)"])
        if dataset_name == "Iris (Çiçek Türleri)":
            data = datasets.load_iris()
        else:
            data = datasets.load_breast_cancer()
        
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.session_state['df'] = df
        st.success(f"{dataset_name} başarıyla yüklendi!")
        
    else:
        file = st.file_uploader("CSV yükleyin", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.session_state['df'] = df
            st.success("Dosya yüklendi!")

    if 'df' in st.session_state:
        st.write("### Veriye İlk Bakış (İlk 5 Satır)")
        st.dataframe(st.session_state['df'].head())


elif app_mode == "Görüntü İşleme":
    st.header("🖼️ Image Pre-processing with NumPy")
    img_file = st.file_uploader("İşlemek için resim seçin", type=['jpg','png'])
    
    if img_file:
        img = Image.open(img_file)
        img_arr = np.array(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Orijinal")
        with col2:
            
            if len(img_arr.shape) == 3:
                red_channel = img_arr.copy()
                red_channel[:, :, 1] = 0 
                red_channel[:, :, 2] = 0 
                st.image(red_channel, caption="Sadece Kırmızı Kanal (NumPy)")


elif app_mode == "Veri Analizi (EDA)":
    st.header("📊 Veri Görselleştirme ve Ön İşleme")
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Betimsel İstatistikler")
            st.write(df.describe())
        with col2:
            st.write("### Özellik Dağılımı")
            feature = st.selectbox("Sütun Seçin", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax, color="purple")
            st.pyplot(fig)
    else:
        st.error("Lütfen önce Ana Sayfadan veri yükleyin!")


elif app_mode == "Boyut İndirgeme & ML":
    st.header("🤖 PCA, Özellik Seçimi ve Model")
    if 'df' in st.session_state:
        df = st.session_state['df'].dropna()
        X = df.drop(columns=['target']) if 'target' in df.columns else df.iloc[:, :-1]
        y = df['target'] if 'target' in df.columns else df.iloc[:, -1]

        
        st.subheader("1. PCA (Boyut İndirgeme)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
        
        fig, ax = plt.subplots()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
        plt.title("PCA ile 2 Boyutlu Gösterim")
        st.pyplot(fig)

        
        st.subheader("2. En İyi Özellikleri Seç (Feature Selection)")
        k = st.slider("Seçilecek özellik sayısı", 1, len(X.columns), 2)
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_names = X.columns[selector.get_support()]
        st.write(f"Seçilen Özellikler: {list(selected_names)}")

        
        if st.button("Modeli Eğit (Random Forest)"):
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            st.success(f"Model Eğitildi! Skor: {model.score(X_test, y_test):.2f}")
            st.text("Rapor:")
            st.text(classification_report(y_test, model.predict(X_test)))