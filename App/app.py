import gradio as gr
import torch
from PIL import Image
from retino_model import Retino
from preprocess_for_inference import preprocess_pil_image

class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

model = Retino(classes=5)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

def predict(image):
    try:
        tensor = preprocess_pil_image(image)
        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()
        return f"Clasa prezisă: {class_names[pred]}"
    except Exception as e:
        return f"Eroare: {str(e)}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Clasificator Retinopatie Diabetică",
    description="Încarcă o imagine fundus (brută sau preprocesată). Modelul aplică automat preprocesare completă."
)

demo.launch()
