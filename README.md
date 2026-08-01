# lstm-next-word-predictor-pytorch[README(1).md](https://github.com/user-attachments/files/30617498/README.1.md)
# LSTM Next-Word Predictor in PyTorch

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![NLP](https://img.shields.io/badge/Task-Next--Word%20Prediction-brightgreen)

A compact **word-level language modelling project** that trains a Long Short-Term Memory (LSTM) neural network to predict the next word in a sequence. The complete workflow—from tokenisation and vocabulary construction to training, evaluation, and autoregressive text generation—is implemented from scratch in PyTorch.

The model is trained on the short narrative **“The Last Train Journey”** and demonstrates how an LSTM can learn local language patterns from sequential text.

## Project Highlights

- Implements a word-level language model using PyTorch
- Tokenises raw text with NLTK
- Builds a custom vocabulary and converts words into integer indices
- Generates prefix-based sequences for supervised next-word prediction
- Uses left-padding to create fixed-length model inputs
- Defines a custom PyTorch `Dataset` and `DataLoader`
- Trains an embedding–LSTM–linear neural network
- Supports single-word prediction and iterative text generation
- Evaluates model performance using training-set accuracy
- Automatically uses a CUDA-enabled GPU when available

## Model Workflow

```text
Raw Text
   ↓
NLTK Word Tokenisation
   ↓
Vocabulary Construction
   ↓
Text-to-Index Conversion
   ↓
Prefix Sequence Generation
   ↓
Sequence Padding
   ↓
Embedding Layer
   ↓
LSTM Layer
   ↓
Fully Connected Layer
   ↓
Next-Word Prediction
```

## Model Architecture

| Component | Configuration |
|---|---|
| Input representation | Word-level token indices |
| Vocabulary size | 284 tokens |
| Embedding layer | 100 dimensions |
| Recurrent layer | Single-layer LSTM |
| LSTM hidden size | 150 units |
| Output layer | Linear layer with 284 outputs |
| Trainable parameters | 222,484 |
| Prediction strategy | Greedy decoding |

The network maps each token to a dense embedding, processes the padded sequence with an LSTM, and uses the final hidden state to predict a probability score for every word in the vocabulary.

## Dataset and Training Configuration

| Setting | Value |
|---|---|
| Training corpus | “The Last Train Journey” |
| Non-empty text lines | 40 |
| Vocabulary size | 284 |
| Training sequences | 555 |
| Maximum sequence length | 25 tokens |
| Model input length | 24 tokens |
| Batch size | 32 |
| Epochs | 50 |
| Learning rate | 0.001 |
| Optimiser | Adam |
| Loss function | Cross-entropy loss |
| Hardware | CPU or CUDA GPU |

Each sentence is transformed into progressively longer prefix sequences. For example:

```text
the last
the last train
the last train journey
```

For each sequence, all tokens except the final token become the input, while the final token becomes the target word.

## Results

The notebook reports:

| Metric | Result |
|---|---:|
| Training accuracy | 96.94% |
| Final epoch cumulative loss | 2.5731 |

> **Important:** The accuracy is calculated on the same sequences used for training. It therefore measures how well the model fits the training corpus and should not be interpreted as validation or test accuracy.

### Example Prediction

```python
prediction(model, vocab, "He chose a seat beside the")
```

Output:

```text
He chose a seat beside the window
```

### Example Text Generation

Starting prompt:

```text
Daniel checked the ticket
```

The model successfully reconstructs:

```text
Daniel checked the ticket in his pocket and walked toward the train.
```

Generation currently uses greedy decoding, meaning the highest-scoring word is selected at every step.

## Repository Structure

```text
lstm-next-word-predictor-pytorch/
├── Language_modeling.ipynb   # Complete preprocessing, training, and inference workflow
└── README.md                 # Project documentation
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/lstm-next-word-predictor-pytorch.git
cd lstm-next-word-predictor-pytorch
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install torch numpy nltk jupyter
```

### 4. Launch the notebook

```bash
jupyter notebook Language_modeling.ipynb
```

Run the cells in order. The notebook downloads the required NLTK `punkt` and `punkt_tab` tokenisation resources automatically.

## Usage

The notebook provides a `prediction` function for predicting one next word:

```python
prediction(
    model=model,
    vocab=vocab,
    text="He chose a seat beside the"
)
```

It can also generate multiple words autoregressively:

```python
num_tokens = 14
input_text = "Daniel checked the ticket"

for _ in range(num_tokens):
    input_text = prediction(model, vocab, input_text)
    print(input_text)
```

## Current Limitations

- The model is trained on a single, small narrative and primarily memorises its patterns.
- No validation or test split is used.
- Token ID `0` is used for both unknown words and sequence padding.
- The prediction function uses a fixed maximum sequence length of 25.
- Greedy decoding can produce repetitive punctuation or repetitive words.
- Model weights, vocabulary, and configuration are not saved for later reuse.
- Random seeds are not fixed, so training results may vary between runs.
- The inference tensor should be moved to the model device when running prediction on a GPU.

## Recommended Improvements

- Train on a larger and more diverse text corpus
- Introduce separate `<pad>`, `<unk>`, `<bos>`, and `<eos>` tokens
- Use `padding_idx` in the embedding layer
- Add training, validation, and test splits
- Track average loss and validation perplexity
- Save and reload model checkpoints and vocabulary files
- Add temperature, top-k, or top-p sampling
- Support variable-length sequences with packed sequences
- Prevent input sequences from exceeding the configured context length
- Refactor the notebook into reusable training and inference modules

## Technologies Used

- Python
- PyTorch
- NLTK
- NumPy
- Jupyter Notebook / Google Colab

## Purpose

This project is intended as an educational demonstration of:

- recurrent neural networks for natural language processing,
- word embeddings,
- sequence preparation,
- next-token classification, and
- autoregressive text generation.

## Acknowledgements

The project uses PyTorch for neural-network development and NLTK for word tokenisation.
