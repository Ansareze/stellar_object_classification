# !pip install gradio

import pandas as pd
import joblib

# 
import gradio as gr

def predict_stellar_object(alpha, delta, u, g, r, i, z, cam_col, redshift, plate, MJD):
    # Create a pandas DataFrame from the input
    input_data = pd.DataFrame({
        'alpha': [alpha],
        'delta': [delta],
        'u': [u],
        'g': [g],
        'r': [r],
        'i': [i],
        'z': [z],
        'cam_col': [cam_col],
        'redshift': [redshift],
        'plate': [plate],
        'MJD': [MJD]
    })

    # Load the saved scaler and model
    loaded_scaler = joblib.load('scaler.joblib')
    loaded_model = joblib.load('gradient_boosting_model.joblib')

    # Scale the input data using the loaded scaler
    input_scaled = loaded_scaler.transform(input_data)

    # Make a prediction
    prediction = loaded_model.predict(input_scaled)

    # Map the numerical prediction back to the class name
    class_mapping = {0: 'GALAXY', 1: 'STAR', 2: 'QSO'}
    predicted_class = class_mapping.get(prediction[0], 'Unknown')

    return predicted_class

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_stellar_object,
    inputs=[
        gr.Number(label="alpha"),
        gr.Number(label="delta"),
        gr.Number(label="u"),
        gr.Number(label="g"),
        gr.Number(label="r"),
        gr.Number(label="i"),
        gr.Number(label="z"),
        gr.Number(label="cam_col"),
        gr.Number(label="redshift"),
        gr.Number(label="plate"),
        gr.Number(label="MJD")
    ],
    outputs=gr.Textbox(label="Predicted Class"),
    title="Stellar Object Classifier",
    description="Predict the class (Galaxy, Star, or Quasar) of a celestial object based on its features."
)

if __name__ == "__main__":
    iface.launch()