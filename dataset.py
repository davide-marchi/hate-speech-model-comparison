import kagglehub

# Download latest version
path = kagglehub.dataset_download("vkrahul/twitter-hate-speech")

print("Path to dataset files:", path)