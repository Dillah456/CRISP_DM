import streamlit as st
import pandas as pd
import pickle

# ==========================================
# Konfigurasi Halaman
# ==========================================
st.set_page_config(
    page_title="Prediksi Risiko Depresi",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# Load Model
# ==========================================
@st.cache_resource
def load_model():
    with open("Mental_Health_Logistic.pkl", "rb") as f:
        model = pickle.load(f)
    return model


# ==========================================
# Main App
# ==========================================
def main():

    model = load_model()
    print(model.feature_names_in_)

    st.title("🧠 Prediksi Risiko Depresi")
    st.markdown(
        """
        Aplikasi ini menggunakan **Logistic Regression**
        untuk memprediksi risiko depresi berdasarkan:

        - Stress Level
        - Sleep Hours
        - Anxiety Level
        - Daily Social Media Hours
        """
    )

    st.divider()

    stress_level = st.slider(
        "Stress Level",
        min_value=1,
        max_value=10,
        value=5
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        min_value=1.0,
        max_value=12.0,
        value=7.0,
        step=0.5
    )

    anxiety_level = st.slider(
        "Anxiety Level",
        min_value=1,
        max_value=10,
        value=5
    )

    daily_social_media_hours = st.slider(
        "Daily Social Media Hours",
        min_value=0.0,
        max_value=15.0,
        value=4.0,
        step=0.5
    )

    st.divider()

    if st.button("Prediksi Risiko Depresi"):

        input_df = pd.DataFrame([{
            "stress_level": stress_level,
            "sleep_hours": sleep_hours,
            "anxiety_level": anxiety_level,
            "daily_social_media_hours": daily_social_media_hours
        }])

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Hasil Prediksi")

        if prediction == 1:
            st.error("⚠️ Berisiko Depresi")
        else:
            st.success("✅ Tidak Berisiko Depresi")

        st.metric(
            label="Probabilitas Depresi",
            value=f"{probability*100:.2f}%"
        )

        with st.expander("Detail Input"):
            st.dataframe(input_df)

        st.info(
            """
            Hasil prediksi merupakan estimasi berdasarkan model machine learning
            dan tidak dapat digunakan sebagai diagnosis medis.
            """
        )


# ==========================================
# Run App
# ==========================================
if __name__ == "__main__":
    main()