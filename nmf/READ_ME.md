## Environment

- Python 3.11.9

## NMF

NMF-EUC와 NMF-DIV를 선택하여 실험할 수 있음.

### NMF-EUC

`scikit-learn`의 `NMF` 기본 설정을 사용하며, Euclidean distance (Frobenius norm)를 최소화함.

```python
model = NMF(
    n_components=J,
    init="nndsvda",
    max_iter=1000,
    random_state=0
)
```

### NMF-DIV

Kullback-Leibler divergence를 최소화하도록 다음과 같이 설정함.

```python
model = NMF(
    n_components=J,
    init="nndsvda",
    solver="mu",
    beta_loss="kullback-leibler",
    max_iter=1000,
    random_state=0
)
```

`beta_loss`를 지정하지 않으면 기본값인 `"frobenius"`가 사용되며, Euclidean distance (Frobenius norm)를 최소화함.