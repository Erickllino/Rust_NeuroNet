# Rust_NeuroNet

Uma rede neural feita **do zero em Rust**, sem nenhuma biblioteca de machine learning — só `rand` para inicializar os pesos. Projeto de aprendizado: o objetivo é entender, na mão, como funcionam forward pass, backpropagation e gradiente descendente.

## Status

- ✅ Operações de matriz próprias (`Matrix`)
- ✅ Camada densa com pesos e bias (`Layer`)
- ✅ Rede com profundidade arbitrária (`NeuralNet` = `Vec<Layer>`)
- ✅ Forward pass iterativo + backpropagation generalizada (qualquer número de camadas)
- ✅ Ativação sigmoide e loss MSE
- ✅ **XOR aprendido** com rede de múltiplas camadas
- 🚧 MNIST (em andamento)

## Estrutura

```
src/
├── main.rs          # ponto de entrada: monta a rede, treina e testa (atualmente XOR)
└── nn/
    ├── mod.rs       # declara os submódulos
    ├── mat.rs       # Matrix: dot, transpose, soma/sub, Hadamard, escalar, etc.
    ├── layer.rs     # Layer: pesos w (n_in × n_out) e bias b (1 × n_out)
    └── nn.rs        # NeuralNet: f_foward, back_prop, loss, train, predict
```

### Convenções de shape

- Entrada: `Matrix(1, n_features)` (uma amostra por vez)
- `Layer.w`: `(n_in, n_out)` — `n_in = w.rows`, `n_out = w.cols`
- `Layer.b`: `(1, n_out)`
- Forward de uma camada: `a = σ(a_prev · w + b)`

## Como rodar

```bash
# debug (lento, bom para desenvolver)
cargo run

# release (otimizado — use para treinos de verdade)
cargo run --release
```

## Como a matemática está organizada

**Forward** (`f_foward`): parte da entrada e, camada a camada, calcula `z = a·w + b` e `a = σ(z)`, propagando `a` adiante.

**Backprop** (`back_prop`): roda um forward guardando `z` e `a` de cada camada, depois percorre as camadas de trás para frente:

- Saída: `δ = (a_L − y) ⊙ σ'(z_L)`
- Escondidas: `δ = (δ · Wₗ₊₁ᵀ) ⊙ σ'(zₗ)`
- Gradientes: `∂W = a_prevᵀ · δ` e `∂b = Σ_linhas(δ)`
- Atualização: `W ← W − lr·∂W`, `b ← b − lr·∂b`

> Importante: o `δ` da camada anterior é calculado **antes** de sobrescrever os pesos da camada atual, para que o gradiente use sempre os pesos antigos.

## Roadmap

- [ ] Carregar MNIST (CSV) com normalização dos pixels (`/255`) e labels one-hot
- [ ] Inicialização Xavier dos pesos (`1/√n_in`) — necessária para a sigmoide não saturar com 784 entradas
- [ ] Avaliação por acurácia (argmax da saída vs. label)
- [ ] Softmax + cross-entropy na saída
- [ ] Mini-batches
- [ ] ReLU nas camadas escondidas

## Dependências

- [`rand`](https://crates.io/crates/rand) — inicialização aleatória dos pesos

Edition 2024.
