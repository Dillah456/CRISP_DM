import gradio as gr
import pandas as pd
import pickle

# Load Model
with open("Mental_Health_Logistic.pkl", "rb") as f:
    model = pickle.load(f)

# Fungsi Prediksi
def predict(stress_level, sleep_hours, anxiety_level, daily_social_media_hours):

    input_df = pd.DataFrame([{
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        "anxiety_level": anxiety_level,
        "daily_social_media_hours": daily_social_media_hours
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        status = "⚠️ Berisiko Depresi"
    else:
        status = "✅ Tidak Berisiko Depresi"

    return f"""
Status: {status}

Probabilitas Depresi: {probability*100:.2f}%
"""

# Interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(1, 10, value=5, step=1, label="Stress Level"),
        gr.Slider(1, 12, value=7, step=0.5, label="Sleep Hours"),
        gr.Slider(1, 10, value=5, step=1, label="Anxiety Level"),
        gr.Slider(0, 15, value=4, step=0.5, label="Daily Social Media Hours")
    ],
    outputs=gr.Textbox(label="Hasil Prediksi"),
    title="🧠 Prediksi Risiko Depresi",
    description="""
Prediksi risiko depresi menggunakan Logistic Regression berdasarkan:

• Stress Level
• Sleep Hours
• Anxiety Level
• Daily Social Media Hours
"""
)

demo.launch()