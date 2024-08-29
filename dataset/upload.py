from huggingface_hub import login
from huggingface_hub import HfApi
api = HfApi()


login(token='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')



api.upload_folder(
    folder_path="/root/blues-1/log",
    repo_id="blue-blue/05000_medical",
    repo_type="model",
)