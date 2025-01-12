import timm
from PIL import Image
import torch
from torchvision import transforms

# Cargar el modelo desde Hugging Face usando timm
model = timm.create_model(
    "hf_hub:CristianR8/vgg19-model",
    num_classes=6,
    in_chans=3,
    pretrained=True,
    exportable=True,
)

# Poner el modelo en modo de evaluación
model.eval()

# Definir una imagen de ejemplo
image_path = "image.png"  # Reemplaza con la ruta de tu imagen
image = Image.open(image_path).convert("RGB")

# Preprocesamiento de la imagen
transform = transforms.Compose([
    transforms.Resize((288, 288)),  # Tamaño de entrada típico para ResNet50
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Valores estándar de normalización para imágenes RGB
        std=[0.229, 0.224, 0.225]
    )
])

# Aplicar las transformaciones
input_tensor = transform(image).unsqueeze(0)  # Añadir una dimensión batch

# Pasar la imagen procesada al modelo
with torch.no_grad():
    output = model(input_tensor)

# Obtener la predicción (índice de la clase con mayor probabilidad)
print(output)
predicted_class = torch.argmax(output, dim=1).item()

print(f"Predicción del modelo: {predicted_class}")
