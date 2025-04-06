import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import defaultdict

# Configurações (devem ser iguais às do treinamento)
local_diretorio = os.path.dirname(os.path.abspath(__file__))
pasta_dos_captchas = os.path.join(local_diretorio, 'captcha-1707')
IMG_WIDTH = 200
IMG_HEIGHT = 50
MAX_TEXT_LENGTH = 8

# Carregar mapeamento de caracteres
def carregar_mapeamentos():
    with open(os.path.join(local_diretorio, 'captchas_labels.json'), 'r') as f:
        captcha_labels = json.load(f)
    
    # Recriar os mapeamentos como no treinamento
    all_chars = set()
    for label in captcha_labels.values():
        all_chars.update(label)
    
    char_to_idx = {char: idx+1 for idx, char in enumerate(sorted(all_chars))}
    char_to_idx[''] = 0  # Para padding
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char

# Definir o modelo (mesma arquitetura do treinamento)
class CRNN(nn.Module):
    def __init__(self, img_height, img_width, num_chars, max_text_length):
        super(CRNN, self).__init__()
        self.max_text_length = max_text_length
        
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        
        # Length predictor
        self.length_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, max_text_length + 1),
            nn.Softmax(dim=1)
        )
        
        # Character predictor
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, max_text_length))
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(256, num_chars)

    def forward(self, x):
        x = self.cnn(x)
        
        # Predict length
        length_logits = self.length_predictor(x)
        
        # Predict characters
        x = self.adaptive_pool(x)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, width, channels * height)
        x, _ = self.rnn(x)
        char_logits = self.fc(x)
        
        return char_logits, length_logits

# Transformações (iguais ao treinamento)
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Função para carregar o modelo treinado
def carregar_modelo_treinado(model_path, char_to_idx, device):
    num_chars = len(char_to_idx)
    model = CRNN(IMG_HEIGHT, IMG_WIDTH, num_chars, MAX_TEXT_LENGTH).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Função para fazer previsões
def prever_captcha(model, image_path, transform, char_to_idx, idx_to_char, device):
    try:
        image = Image.open(image_path).convert('L')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            char_logits, length_logits = model(image)  # Recebe ambos os retornos
            char_logits = char_logits.permute(0, 2, 1)
            
            # Obter comprimento previsto
            predicted_length = torch.argmax(length_logits).item()
            
            # Obter caracteres previstos
            _, predicted = torch.max(char_logits, 1)
            predicted = predicted.squeeze(0)
            
            text = []
            for idx in predicted[:predicted_length]:  # Cortar no comprimento previsto
                if idx.item() != 0:
                    text.append(idx_to_char[idx.item()])
            
            return ''.join(text)
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return None


# Adicione esta nova função após a função prever_captcha
def rodar_modelo_iniciar(model, image_path, transform, char_to_idx, idx_to_char, device):
    # Carregar os labels verdadeiros se necessário
    with open(os.path.join(local_diretorio, 'captchas_labels.json'), 'r') as f:
        captcha_labels = json.load(f)
    
    nome_arquivo = os.path.basename(image_path)
    predicao = prever_captcha(model, image_path, transform, char_to_idx, idx_to_char, device)
    
    verdadeiro = captcha_labels.get(nome_arquivo, "DESCONHECIDO")
    acerto = (predicao == verdadeiro)
    
    print(f"\nResultado para {nome_arquivo}:")
    print(f"Predição: {predicao}")

    resultado = str(predicao)
    
    return resultado

# Ex: pra iniciar e rodar
def main():

    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Carregar mapeamentos
    char_to_idx, idx_to_char = carregar_mapeamentos()
    print(f"Carregados mapeamentos para {len(char_to_idx)} caracteres")
    
    # Carregar modelo treinado
    model_path = os.path.join(local_diretorio, 'modelo-M02.pth')
    if not os.path.exists(model_path):
        print("Erro: Arquivo do modelo 'modelo-M02.pth' não encontrado!")
        return
    
    model = carregar_modelo_treinado(model_path, char_to_idx, device)
    print("Modelo carregado com sucesso!")
    
    # Aqui ele teria que pegar a imagem no site, coloca 
    caminho_imagem = os.path.join(local_diretorio, 'captcha', 'captcha_28.png')
    
    # Aqui roda o modelo
    rodar_modelo_iniciar(model, caminho_imagem, transform, char_to_idx, idx_to_char, device)

if __name__ == "__main__":
    main()