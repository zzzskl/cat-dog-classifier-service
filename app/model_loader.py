import torch
import torch.nn as nn
import torchvision.models as models
import os
import wandb

PROJECT_NAME = "mlops-image-classifier"
ARTIFACT_NAME = "cat-dog-classifier"
ARTIFACT_ALIAS = "latest"
NUM_CLASSES = 2
MODEL_FILE_NAME = "resnet_cat_dog_filtered_production_v3_best.pth" # Corrected filename based on our training script

def _setup_model_architecture(num_classes):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def load_model():
    print("Initializing W&B run to download model...")
    # Use anonymous='allow' to avoid login prompts in a server environment
    run = wandb.init(project=PROJECT_NAME, job_type="download-model", anonymous="allow")
    
    try:
        print(f"Downloading artifact: {ARTIFACT_NAME}:{ARTIFACT_ALIAS}")
        artifact = run.use_artifact(f'{ARTIFACT_NAME}:{ARTIFACT_ALIAS}', type='model')
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, MODEL_FILE_NAME)

        print(f"Model downloaded to: {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = _setup_model_architecture(NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        print(f"Model loaded successfully on device: {device}")
        return model, device

    finally:
        print("Finishing W&B run.")
        run.finish()
