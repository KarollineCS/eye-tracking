# Projeto de Gaze-Tracking

## 🛠️ Instalação

1. **Clone o repositório**
```bash
git clone https://github.com/KarollineCS/eye-tracking.git
cd eye-tracking
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```
*Opte por fazer a instalação das dependências em Ambiente Virtual para não ocorrer nenhum conflito de versões*

3. **Execute o sistema**
```bash
python src/main.py
```

---

## 🎮 Modo de usar

1. Pressione `c` para iniciar a **calibração** na tela de visualização da câmera
2. Olhe para cada ponto vermelho e pressione `SPACE`
3. Após calibração, o sistema **predirá** onde você está olhando
4. Use `r` para resetar calibração e recalibrar se necessário

**Controles adicionais**
- `l`: alternar landmarks
- `i`: alternar detecção de íris
- `q`: sair

---
*Baseado no artigo: 'Efficiency in Real-time Webcam Gaze Tracking*
