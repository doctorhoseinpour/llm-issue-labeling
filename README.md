# Pipeline Setup Instructions

## 1. Clone the repo and set up the environment

```bash
git clone <REPO_URL>
cd <REPO_NAME>
python3 -m venv ragtag_env
source ragtag_env/bin/activate
pip install -r requirements.txt
```

## 2. Add the dataset files

Copy the two CSV files I sent you into the repo root directory:

```
issues3k.csv
issues30k.csv
```

## 3. Make the scripts executable

```bash
chmod +x run_full_test.sh run_pipeline.sh
```

## 4. Run the pipeline

```bash
./run_full_test.sh
```
