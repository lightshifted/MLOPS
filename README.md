# Setup

## Environment

```python
python -m venv venv
& venv/scripts/activate
python -m pip install pip setuptools wheel
python -m pip install -e .
```

## Model Serving

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```

## Git Branch Merge

```bash
git checkout master
git branch main master -f
git checkout main
git push origin main -f
```

## Generate Pre-Commit YAML (POWERSHELL)

```bash
pre-commit sample-config | out-file .pre-commit-config.yaml -encoding utf8
```
