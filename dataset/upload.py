from huggingface_hub import login

login(token='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')



api.upload_folder(
    folder_path="dataset/HiTZ/Multilingual-Medical-Corpus",
    repo_id="blue-blue/medical_dataset_shards",
    repo_type="dataset",
)