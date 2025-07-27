# Stable Diffusion - Electrical Engineering Image Generator

Este projeto utiliza o modelo Stable Diffusion (`runwayml/stable-diffusion-v1-5`) para gerar imagens relacionadas à engenharia elétrica, como laboratórios com circuitos, osciloscópios, multímetros e transformadores.

## ✅ Requisitos

Certifique-se de ter uma GPU com suporte a CUDA 12.1 (ou ajuste conforme seu hardware) e Python 3.8+.

### Instalação

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors
````

### Autenticação (se necessário)

Você precisará de uma conta no [Hugging Face](https://huggingface.co/) e executar:

```bash
huggingface-cli login
```

## ▶️ Execução

```bash
python main.py
```

A imagem será salva como `engenharia_eletrica.png` no diretório atual.

## 🎯 Exemplo de prompt usado

> *An electrical engineering laboratory, with circuits, oscilloscopes, multimeters and transformers, detailed wires, soft lighting, highly realistic, digital art style, futuristic atmosphere.*

## 📁 Estrutura do Projeto

```
sd/
├── main.py
├── README.md
```

## 🧠 Créditos

* Modelo: [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
* Diffusers Library: [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

````

---

### 📄 `main.py`

```python
from diffusers import StableDiffusionPipeline
import torch

# Definir prompt
prompt = (
    "An electrical engineering laboratory, with circuits, oscilloscopes, multimeters and transformers, "
    "detailed wires, soft lighting, highly realistic, digital art style, futuristic atmosphere"
)

# Carregar modelo
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Gerar imagem
image = pipe(prompt).images[0]

# Salvar imagem
image.save("engenharia_eletrica.png")
print("Imagem gerada e salva como engenharia_eletrica.png")
````

