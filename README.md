# Weather RNN Classifier

A PyTorch-based Recurrent Neural Network (RNN) for multi‑label weather condition classification using time‑series meteorological data. The model learns from sliding‑window sequences and predicts 27 possible weather states.

## Features
- Sequence-based dataset (`SequencedDataset`)
- Custom RNN model with dropout + L2 regularization
- Class imbalance handling via `pos_weight`
- Early stopping (patience + threshold)
- Evaluation mode with baseline metrics

## Data
The script expects a CSV at:

```
processed_data/final.csv
```

### Feature Columns
Temperature, humidity, wind direction/speed, visibility, station pressure, wind chill, time encodings, year.

### Target Columns
27 weather condition labels (e.g., Rain, Fog, Snow, Thunderstorms).

## Training
Set:

```python
EVALUATION_MODE = False
```

Run:

```
python main.py
```

The best model is saved to:

```
model/rnn_model.pt
```

## Evaluation
Set:

```python
EVALUATION_MODE = True
```

Run:

```
python main.py
```

Outputs:
- Zero/one baselines  
- Most-common-vote baseline  
- Model accuracy, recall, precision, F1  

## Model
- RNN (3 layers, hidden size 256, dropout 0.3)
- Linear output → 27 logits
- Sigmoid applied manually during prediction

## File Structure
- `main.py` — training + evaluation logic  
- `model/rnn_model.pt` — saved weights  
- `processed_data/final.csv` — input dataset  
