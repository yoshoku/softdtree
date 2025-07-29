# softdtree

softdtree is a Python library that implements classifier and regressor with Soft Decision Tree.

## Installation

softdtree requires Eigen3, so install it beforehand,

macOS:

```bash
$ brew install eigen cmake
```

Ubuntu:

```bash
$ sudo apt-get install libeigen3-dev cmake
```

Then, install softdtree from [PyPI](https://pypi.org/project/softdtree):

```bash
$ pip install -U softdtree
```

## Usage

The API of softdtree is compatible with [scikit-learn](https://scikit-learn.org/stable/).

### Classifier

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from softdtree import SoftDecisionTreeClassifier

X, y = load_digits(n_class=4, return_X_y=True)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("tree", SoftDecisionTreeClassifier(
        max_depth=4, eta=0.01, max_epoch=100, random_seed=42)),                                                                           ])

scores = cross_val_score(clf, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Regressor

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from softdtree import SoftDecisionTreeRegressor

X, y = load_diabetes(return_X_y=True)

reg = Pipeline([
    ("scaler", MinMaxScaler()),
    ("tree", SoftDecisionTreeRegressor(
        max_depth=4, eta=0.1, max_epoch=100, random_seed=42)),
])

scores = cross_val_score(reg, X, y, cv=5)
print(f"R^2: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Properties

## References

- O. Irsoy, O. T. Yildiz, and E. Alpaydin, "Soft Decision Trees," In Proc. ICPR2012, 2012.
- O. Irsoy and E. Alpaydin, "Dropout regularization in hierarchical mixture of experts," Neurocomputing, vol. 419, pp. 148--156, 2021.

## License

softdtree is available as open source under the terms of
the [BSD-3-Clause License](https://github.com/yoshoku/softdtree/blob/main/LICENSE.txt).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yoshoku/softdtree
This project is intended to be a safe, welcoming space for collaboration,
and contributors are expected to adhere to the [Contributor Covenant](https://contributor-covenant.org) code of conduct.
