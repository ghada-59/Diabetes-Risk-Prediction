# Contributing to Diabetes Risk Prediction

Thank you for contributing to this medical ML project.

## Getting started

```bash
git clone https://github.com/ghada-59/Diabetes-Risk-Prediction.git
cd Diabetes-Risk-Prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project scope

This repository focuses on reproducible diabetes risk modeling, including:
- data cleaning and preprocessing,
- model comparison,
- evaluation metrics,
- clinical interpretation,
- Streamlit-based decision support interface.

## Code standards

- Keep functions modular and documented.
- Prefer clear variable names and short, readable pipeline steps.
- Add or update tests when changing model logic or preprocessing.
- Use sklearn pipelines when training models to avoid data leakage.

## Validation before opening a PR

```bash
python src/data_loader.py
python src/preprocessing.py
python src/train.py
python src/evaluate.py
```

If your change affects the app UI, also run:

```bash
streamlit run app.py
```

## Pull request workflow

1. Fork the repository or create a branch.
2. Commit with a clear message.
3. Run the validation commands above.
4. Open a pull request describing the clinical or technical improvement.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
