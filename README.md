# CNN_comparing

Modelagem de uma CNN para comparar com uma rede reconhecida (ResNet50), usando PyTorch.

## Estrutura

- `/src/cnn_model.py`: classe da CNN customizada e constantes de otimização/regularização.
- `/scripts/run_experiments.py`: treino e avaliação em **MNIST** e **CIFAR10** para CNN customizada e ResNet50.
- `/scripts/plot_results.py`: geração de gráficos comparativos.
- `/notebooks/experimentos.ipynb`: execução dos experimentos + visualização de ativações de kernels.
- `/notebooks/resultados.ipynb`: análise consolidada + seção de questões para justificar os resultados.

## Como executar

```bash
pip install -r requirements.txt
python scripts/run_experiments.py --epochs 5 --batch-size 64 --output-dir results
python scripts/plot_results.py --results-dir results --output-dir results/plots
```

Os artefatos serão gerados em `results/`:
- `summary.json`
- `training_history.csv`
- `plots/*.png`

## Técnicas de regularização aplicadas

- Dropout / Dropout2d
- Batch Normalization
- Weight Decay (L2)
- Data augmentation leve (rotação para MNIST e flip/crop para CIFAR10)
