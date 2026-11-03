import os
from faster_whisper.utils import download_model

# Set via the Dockerfile's MODEL_NAME build arg; must match the MODEL_NAME app.py loads.
model_name = os.getenv("MODEL_NAME", "turbo")

print(f"Downloading {model_name}...")
download_model(model_name, cache_dir=None)
print(f"Finished downloading {model_name}.")
