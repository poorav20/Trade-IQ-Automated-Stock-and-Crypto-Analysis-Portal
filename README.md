# TradeIQ

TradeIQ is a local-first stock and cryptocurrency analysis portal with a Streamlit dashboard, market-data ingestion, ML signals, news sentiment, paper-trading tools, and an optional n8n automation bridge.

## Quick Start

Requirements: Python 3.10 or newer. From the repository directory, create and activate a virtual environment, install dependencies, and copy the example environment file:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` only for integrations you use. Keep live broker mode disabled unless you have explicitly configured and tested the broker credentials. Then initialize the local database, fetch market data, and start the dashboard:

```powershell
python database.py
python data_fetcher.py
python ml_model_v2.py
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Data providers may limit or change access; the dashboard requires an internet connection for live quotes and news.

## Historical CSV Data

Historical data is optional. Put OHLCV CSV files in `data/raw/`; `load_and_train.py` and `train_agents.py` discover files there. Then run the ingestion/training command you need:

```powershell
python load_and_train.py
# or
python train_agents.py
```

Raw datasets are excluded from Git because they can be large and may have separate redistribution terms. The database and training output are local runtime files and are excluded too.

## n8n Automation

The local workflow is [`workflows/n8n/n8n_workflow_FINAL_LOCAL.json`](workflows/n8n/n8n_workflow_FINAL_LOCAL.json). It calls the FastAPI bridge at `127.0.0.1:8502`, refreshes data, runs sentiment and predictions, and performs a paper-trading cycle. n8n and Node.js must be installed separately.

Start both local services with:

```powershell
.\start_local_automation.bat
```

Or start `python webhook_server.py` and `npx n8n` separately. In n8n, import the workflow JSON, execute it once manually, and activate its schedule only after reviewing the workflow. `launch_n8n.bat` adjusts n8n's outgoing-IP restriction so local HTTP requests can reach the bridge; use it only for trusted local workflows.

FastAPI documentation is available at [http://localhost:8502/docs](http://localhost:8502/docs) while the bridge is running. The mobile launcher and ngrok flow expose the dashboard beyond localhost; use them only on trusted networks and do not publish private tunnel URLs.

## Repository Layout

```text
app.py, database.py, *.py     Dashboard, data, models, and service code
data/raw/                     Optional local historical CSV inputs (ignored)
workflows/n8n/                n8n workflow exports
docs/                         Methodology, papers, and project notes
docs/deliverables/            Project reports and presentations
artifacts/figures/             Generated charts (ignored)
local_only/                    Private files and local reference projects (ignored)
```

## Configuration and Safety

- Copy `.env.example` to `.env`; never commit `.env`, API keys, access tokens, or broker credentials.
- `LIVE_MODE` defaults to `False`. This project is for analysis and paper trading by default, not financial advice.
- Local SQLite databases, raw datasets, generated charts, logs, and installers are excluded by `.gitignore`.
- Streamlit binds to `localhost` by default. `launch_mobile.bat` deliberately enables network access for its mobile workflow.

## Documentation

- [Methodology](docs/methodology.md)
- [Project report](docs/TradeIQ_Project_Report.md)
- [IEEE paper](docs/TradeIQ_IEEE_Paper.md)
- [Project outcomes](docs/outcomes.md)

## Publish to GitHub

Review the files before staging. Raw datasets, credentials, local databases, generated charts, and private files are ignored, but check the staged list before committing:

```powershell
git status --short
git add .
git status --short
git commit -m "Prepare TradeIQ project"
git branch -M main
git remote add origin https://github.com/<your-account>/<your-repository>.git
git push -u origin main
```
