from torchvision import transforms
from PIL import Image
import io
import torch

# --- Configuration ---
CLASS_NAMES = ['cat', 'dog']

# --- Preprocessing ---
# This transform must be identical to the one used for validation/testing
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_from_bytes(model, device, image_bytes: bytes) -> dict:
    """
    Takes a model, device, and image in bytes format, and returns
    a prediction dictionary.
    """
    try:
        # Convert bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply transformations and add a batch dimension
        image_tensor = inference_transforms(image).unsqueeze(0)
        
        # Move tensor to the correct device
        image_tensor = image_tensor.to(device)
        
        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
            
        # Get probabilities and find the max
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, 0)
        
        prediction = CLASS_NAMES[predicted_index.item()]
        
        return {
            "prediction": prediction,
            "confidence": confidence.item()
        }

    except Exception as e:
        # In a real application, you'd want more robust error logging here
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

