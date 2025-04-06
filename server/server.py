from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- IMPORTANTE
from runmodelo import carregar_modelo_treinado, transform, carregar_mapeamentos, rodar_modelo_iniciar
import os
import base64
from io import BytesIO
from PIL import Image
import torch

app = Flask(__name__)
CORS(app)  # <-- ATIVA CORS para todas as origens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_diretorio = os.path.dirname(os.path.abspath(__file__))

# Inicializa o modelo
char_to_idx, idx_to_char = carregar_mapeamentos()
model_path = os.path.join(local_diretorio, 'modelo-M02.pth')
model = carregar_modelo_treinado(model_path, char_to_idx, device)

@app.route('/resolver-captcha', methods=['POST'])
def resolver_captcha():
    data = request.get_json()
    imagem_base64 = data.get('imagem')

    if not imagem_base64:
        return jsonify({'erro': 'Imagem não fornecida'}), 400

    try:
        # Converte base64 -> imagem
        image_data = base64.b64decode(imagem_base64)
        image = Image.open(BytesIO(image_data)).convert('L')
        
        # Salva temporariamente
        temp_path = os.path.join(local_diretorio, 'temp_captcha.png')
        image.save(temp_path)

        # Roda o modelo
        resultado = rodar_modelo_iniciar(model, temp_path, transform, char_to_idx, idx_to_char, device)

        return jsonify({'resultado': resultado})
    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
