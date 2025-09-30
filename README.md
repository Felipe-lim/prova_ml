### Reconhecimento de Sinais em Libras a partir de Keyframes de Pose

Este repositório reúne experimentos de aprendizado de máquina para classificar e agrupar sinais em Libras (Língua Brasileira de Sinais) a partir de keyframes extraídos de vídeos. O trabalho está organizado em dois notebooks principais: um de classificação supervisionada e outro de clusterização não supervisionada.

Referência de dados: veja o documento `data/Descrição do Corpus de Keyframes em Libras.pdf` e os arquivos em `data/Sinais/`.

---

### Visão Geral do Problema
- **Entrada**: arquivos JSON contendo, por frame, keypoints de pose (x, y, z, visibility). Cada JSON representa a execução de um sinal por um intérprete.
- **Objetivo 1 (notebook 01)**: classificar o sinal (25 classes) com generalização entre intérpretes diferentes.
- **Objetivo 2 (notebook 02)**: explorar a estrutura dos dados via clusterização e projeções 2D, avaliando a separabilidade intrínseca entre classes.

---

### Dados e Extração de Atributos
- Fonte tabular: `data/Sinais/sinais.csv` (metadados: `file_name`, `width`, `height`, `duration_sec`, `num_frames`, `sinal`, `interprete`).
- Arquivos JSON por amostra em `data/Sinais/Sinais/`.
- Pré-processamento por amostra (em ambos os notebooks):
  - Normalização das coordenadas usando o keypoint id=0 do frame 0 como referência (centralização).
  - Seleção de keypoints do tronco superior e membros superiores (IDs 11–22 no notebook de classificação; 13–22 no notebook de clusterização).
  - Cálculo de estatísticas por keypoint: média, desvio padrão, mínimo e máximo de x e y (e visibility durante a construção, depois descartada na classificação).
  - Geração opcional de data augmentation com pequenas perturbações aleatórias (±5%) apenas em x e y, multiplicando o conjunto (original + 3 variantes).
- Filtragem de qualidade: remoção de amostras com visibilidade média < 0,5 em qualquer keypoint considerado.
- Após a construção e limpeza, o conjunto consolidado tem ~9.960 amostras (≈ 10k) e de 80 a 100+ atributos numéricos, antes de reduções adicionais.

---

### Notebook 01 — Classificação Supervisionada (`notebooks/01_classificacao.ipynb`)
Pipeline principal:
- Remoções: colunas de `visibility_*` e metadados (`file_name`, `duration_sec`, `num_frames`, `width`, `height`).
- Alvo: `sinal` (codificado via `LabelEncoder`).
- Split por intérprete para evitar vazamento: teste em `['Dannubia', 'Cecilia']`. Em variação posterior, validação em `['Jackeline']` para early stopping.
- Escalonamento: `RobustScaler` em features numéricas.
- Redução de dimensionalidade: `PCA(n_components=40)` em alguns modelos.
- Validação: `StratifiedGroupKFold` (agrupando por `interprete`) com métrica principal `F1-macro`.

Modelos avaliados e resultados (F1-macro no teste entre intérpretes):
- Random Forest (300 árvores, `max_depth=20`, `min_samples_split=10`): ≈ **0,496**
- MLP (pipeline com PCA): ≈ **0,469**
- MLP (treino com early stopping usando validação por intérprete): ≈ **0,399**
- K-NN (k=3, `weights=distance`, métrica Manhattan): ≈ **0,397**

Outras saídas disponíveis no notebook:
- Relatórios de classificação detalhados por classe e matrizes de confusão.
- Curva de loss do MLP e validação cruzada com grupos.

Observações:
- A divisão por intérprete é crucial para medir generalização real (evita que o modelo memorize o estilo do intérprete).
- A remoção de `visibility_*` ajudou a reduzir ruído; componentes latentes via PCA melhoraram estabilidade de alguns modelos.

---

### Notebook 02 — Clusterização e Projeções 2D (`notebooks/02_clusterizacao.ipynb`)
Preparação e visualização:
- `SimpleImputer(mean)` + `StandardScaler` sobre as features numéricas (sem o rótulo `sinal`).
- Projeções 2D para inspeção visual: **PCA**, **t-SNE** e **UMAP** (opcional, instalado via `umap-learn`).

K-means e escolha de K:
- Curva do cotovelo (inércia) e `silhouette` ao varrer K.
- Heurística do cotovelo sugeriu `K ≈ 12` (para referência; pode-se também fixar K=25 para comparar com o número de classes reais).

Modelos e avaliação:
- K-means (K do cotovelo) e Agglomerative Clustering (linkages `ward` e `average`).
- Métricas internas: `silhouette`, `calinski_harabasz`, `davies_bouldin`.
- Métricas externas (apenas para análise, usando rótulos reais): `ARI`, `NMI`, `homogeneity`, `completeness`, `v-measure` e **purity** (implementada no notebook).
- Plots dos clusters nas projeções (cores por cluster) e das classes (cores por rótulo) para contraste visual.

Principais achados qualitativos:
- As projeções mostram agrupamentos razoáveis para algumas classes, mas há sobreposição significativa — consistente com os resultados de classificação.
- O melhor linkage/algoritmo depende do K e da métrica escolhida; `ward` costuma favorecer estruturas mais compactas em Euclidiano.

---

### Como Reproduzir
1) Criar ambiente e instalar dependências (Python 3.11 recomendado):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Abrir os notebooks:
- Execute as células em `notebooks/01_classificacao.ipynb` e `notebooks/02_clusterizacao.ipynb`.

Notas de execução:
- Alguns JSONs no dataset estao ausentes ou com keypoint id=0 no frame 0; o código já ignora tais casos e registra mensagens.
- `umap-learn` é opcional; caso não carregue, o notebook prossegue sem UMAP.

---

### Estrutura do Repositório (essencial)
- `data/` — PDF de referência e corpus (`Sinais/`, `sinais.csv`, JSONs por amostra).
- `notebooks/` — experimentos de classificação (01) e clusterização (02).
- `src/` — espaço reservado para modularização futura de código reutilizável.
- `requirements.txt` — versões fixadas para reprodutibilidade.

---

### Limitações e Próximos Passos
- Não utilizamos o eixo z (não mostrou ganhos nos testes iniciais); explorar 3D e/ou profundidade pode ajudar.
- As features são majoritariamente estatísticas por keypoint (média, std, min, max). Incluir dinâmica temporal (velocidades, acelerações, deltas inter-frame) pode capturar melhor a semântica do gesto.
- Aumento de dados mais rico (rotação/escala coerentes com o corpo, jitter controlado por parte do corpo) e técnicas de regularização podem melhorar generalização.
- Avaliar arquiteturas sequenciais (RNN/Temporal CNN/Transformers) com séries de keypoints, em vez de apenas estatísticas agregadas.
- Investigar landmarks de mãos/dedos com maior resolução (ex.: MediaPipe Hands) para sinais com sutilezas manuais.
- Harmonizar a seleção de IDs de keypoints entre notebooks (11–22 vs. 13–22) e revisar sensibilidade a essa escolha.

---
