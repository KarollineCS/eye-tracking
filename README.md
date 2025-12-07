# Sistema de Eye-Tracking de Baixo Custo para Análise de Consciência Situacional

**Este projeto possui propósito exclusivamente acadêmico, sendo desenvolvido como Trabalho de Conclusão de Curso (TCC) em Engenharia de Computação.**

---

## 📋 Descrição

Sistema de rastreamento ocular (*eye-tracking*) baseado em webcam convencional, desenvolvido para análise de consciência situacional em simuladores de mineração. O sistema utiliza técnicas de visão computacional e aprendizado de máquina para estimar a direção do olhar sem necessidade de hardware especializado.

### Principais características:
- Detecção de 468 landmarks faciais via MediaPipe
- Modelagem 3D de esferas oculares
- Compensação de pose da cabeça (invariância de pose) via PCA
- Calibração por Regressão Polinomial com regularização Ridge
- Validação através de protocolo Chessboard
- Exportação de métricas para análise offline

---

## 👥 Autores

**Discente:**
- Karolline Carvalho Silva — [karollinecarvalhosilva@gmail.com](mailto:karollinecarvalhosilva@gmail.com)

**Orientador:**
- Prof. Dr. Giovani Bernardes Vitor
  
**Coorientadora:**
- Natasha Sayuri Dias Nakashima — Doutoranda, UNICAMP

**Instituição:**
- Universidade Federal de Itajubá (UNIFEI) — Campus Itabira

---

## 🛠️ Instalação

### Pré-requisitos
- Python 3.8 ou superior
- Webcam funcional

> 💡 **Dica:** Você pode usar a câmera do celular como webcam através de aplicativos como [DroidCam](https://www.dev47apps.com/) (Android/iOS). Isso geralmente oferece melhor qualidade de imagem do que webcams integradas em notebooks.

### Passos

1. **Clone o repositório**
```bash
git clone https://github.com/KarollineCS/eye-tracking.git
cd eye-tracking
```

2. **Crie um ambiente virtual (recomendado)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Execute o sistema**
```bash
python main.py
```

---

## 🎮 Modo de Usar

### Inicialização
1. Execute `python main.py`
2. Posicione-se de frente para a webcam (~50-60 cm de distância)
3. Certifique-se de boa iluminação ambiente

### Calibração
1. Pressione `e` para calibrar as **esferas oculares** (mantenha a cabeça parada)
2. Pressione `p` para calibrar o **plano do monitor**
3. Pressione `h` para executar a **calibração de 9 pontos**:
   - Olhe para cada ponto vermelho exibido na tela
   - Pressione `SPACE` para confirmar cada ponto

### Validação
- Pressione `y` para iniciar o **modo de validação Chessboard**
- O sistema irá avaliar a precisão em uma grade 4×3

### Controles

**Calibração:**
| Tecla | Função |
|-------|--------|
| `e` | Calibrar esferas oculares |
| `o` | Calibrar offset de 1 ponto (sistema 3D) |
| `h` | Calibração de 9 pontos |
| `r` | Resetar calibração |

**Validação e Visualização:**
| Tecla | Função |
|-------|--------|
| `y` | Iniciar modo de validação Chessboard |
| `v` | Alternar visualização do alvo 3D |
| `t` | Alternar entre sistema 2D e 3D |
| `i` | Mostrar informações 3D |
| `d` | Ativar/desativar modo debug |

**Filtros e Suavização:**
| Tecla | Função |
|-------|--------|
| `k` | Ativar/desativar filtro Kalman |
| `s` | Ativar/desativar suavização de gaze |
| `]` | Aumentar buffer de suavização |
| `[` | Diminuir buffer de suavização |
| `+` | Aumentar ganho do gaze |
| `-` | Diminuir ganho do gaze |

**Ajustes:**
| Tecla | Função |
|-------|--------|
| `c` | Ativar/desativar clamping (limitar às bordas) |
| `z` | Inverter eixo X |
| `↑` | Ajustar monitor para cima |
| `↓` | Ajustar monitor para baixo |
| `g` | Mostrar configurações de gaze |
| `m` | Mostrar métricas do Kalman |

**Sistema:**
| Tecla | Função |
|-------|--------|
| `q` | Sair e salvar dados |

> ⚠️ **Nota sobre suavização:** Recomenda-se usar apenas um método de suavização por vez (Kalman OU buffer de média móvel). O uso simultâneo pode causar *drift* (deriva) no cursor.

---

## 📊 Resultados

O sistema alcançou os seguintes resultados na validação, sem aplicação do filtro Kalman:

| Métrica | Valor |
|---------|-------|
| Taxa de acerto (AOIs) | 75,1% |
| Erro médio | 184,0 pixels |
| Erro Angular Médio (MAE) | ~4,8° |

---

## 📁 Estrutura do Projeto

```
eye-tracking/
├── main.py                     # Ponto de entrada principal
├── requirements.txt            # Dependências
├── README.md
│
└── src/
    ├── config/
    │   └── settings.py         # Configurações do sistema
    │
    ├── core/
    │   ├── face_detector.py    # Detecção facial
    │   ├── iris_tracker.py     # Rastreamento de íris
    │   └── gaze_calculator.py  # Cálculo do vetor de gaze
    │
    ├── calibration/
    │   ├── screen_calibration.py
    │   ├── gaze_to_screen_3d.py
    │   ├── advanced_calibration.py
    │   ├── chessboard_validator.py
    │   └── refinement_calibration.py
    │
    ├── filters/
    │   ├── kalman_filter.py    # Filtro de Kalman
    │   └── binocular_fusion.py # Fusão binocular
    │
    └── utils/
        ├── performance.py
        ├── visualization.py
        └── data_logger.py      # Exportação de dados
```

---

## 🔗 Referências

### Artigos Científicos

- **Gudi, A., Li, X., van Gemert, J.** (2020). *Efficiency in Real-Time Webcam Gaze Tracking*. In: Bartoli, A., Fusiello, A. (eds) Computer Vision – ECCV 2020 Workshops. Lecture Notes in Computer Science, vol 12535. Springer, Cham. https://doi.org/10.1007/978-3-030-66415-2_34

- **Endsley, M. R.** (1995). *Toward a Theory of Situation Awareness in Dynamic Systems*. Human Factors, 37(1), 32-64.

- **Holmqvist, K. et al.** (2011). *Eye Tracking: A Comprehensive Guide to Methods and Measures*. Oxford University Press.

- **Lugaresi, C. et al.** (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv:1906.08172.

### Repositórios Base

- **MonitorTracking** (Jason Orlosky) — Implementação base da modelagem 3D de esferas oculares e estimativa de pose via PCA.
  - https://github.com/JEOresearch/MonitorTracking

---

## 📄 Licença

Este projeto é de uso acadêmico. Para outros fins, entre em contato com os autores.

---

## 🙏 Agradecimentos

- À UNIFEI Campus Itabira pela infraestrutura
- Ao Prof. Dr. Giovani Bernardes Vitor pela orientação
- À Natasha Sayuri pela coorientação e contexto da pesquisa de doutorado
- Aos desenvolvedores do MediaPipe e das bibliotecas open-source utilizadas
