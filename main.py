!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install diffusers transformers accelerate safetensors

from diffusers import StableDiffusionPipeline
import torch

# Carregar pipeline com modelo base
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
).to("cuda")

# Prompt: Engenharia Elétrica
prompt = "An electrical engineering laboratory, with circuits, oscilloscopes, multimeters and transformers."

# Gerar imagem
image = pipe(prompt).images[0]

# Salvar ou exibir
image.save("engenharia_eletrica.png")
image.show()
