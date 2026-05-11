# TradeIQ – Automated Trading Portal

[![GitHub license](https://img.shields.io/github/license/your‑username/tradeiq)](LICENSE)  
[![GitHub stars](https://img.shields.io/github/stars/your‑username/tradeiq?style=social)](https://github.com/your-username/tradeiq/stargazers)  
[![GitHub issues](https://img.shields.io/github/issues/your‑username/tradeiq)](https://github.com/your-username/tradeiq/issues)  

---  

## Table of Contents  

1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Architecture Diagram](#architecture-diagram)  
4. [Installation](#installation)  
5. [Configuration](#configuration)  
6. [Running the Application](#running-the-application)  
7. [Demo & Screenshots](#demo--screenshots)  
8. [Testing](#testing)  
9. [Contribution Guide](#contribution-guide)  
10. [Roadmap](#roadmap)  
11. [Citing the Project](#citing-the-project)  
12. [License](#license)  
13. [Acknowledgements](#acknowledgements)  

---  

## Project Overview  

**TradeIQ** is a modern, **paper‑trading** portal built with **Streamlit** that combines **LLM‑driven decision making**, **technical‑indicator analysis**, and a **risk‑management layer**. It enables users to:

* Simulate multi‑asset trading strategies in real‑time.  
* Visualize sentiment, price‑action, and risk metrics with interactive charts.  
* Deploy custom Python‑based plugins for new data sources or broker APIs.  

The repository follows a **clean modular architecture** inspired by the Hyperliquid‑trading‑agent design, allowing easy swapping of the LLM, risk manager, or indicator suite.

---  

## Key Features  

| Feature | Description |
|---------|-------------|
| **LLM Agent** | Uses OpenAI / Gemini APIs (configurable) to generate trade signals from market context and user prompts. |
| **Technical Indicators** | Built‑in SMA, EMA, RSI, MACD, VADER‑sentiment, and custom indicator plug‑ins. |
| **Risk Management** | Position sizing, stop‑loss/take‑profit, portfolio‑level VaR checks, and hysteresis‑based guardrails. |
| **Paper‑Trading Engine** | Event‑driven loop that executes simulated orders against a SQLite‑backed `trading.db`. |
| **Streamlit UI** | Real‑time dashboards: order book, trade log, equity curve, heat‑maps, and sentiment timeline. |
| **Data Generation** | Utility scripts (`generate_demo_data.py`, `generate_paper_graphs.py`) for fast prototyping. |
| **Extensible Plugin System** | Drop a `*.py` file into `plugins/` to add new data feeds or broker adapters without touching core code. |
| **CI/CD** | GitHub Actions workflow for linting, testing, and automatic Docker image publishing. |

---  

## Architecture Diagram  

```mermaid
graph TD
    A[Streamlit Front‑end] --> B[API Router]
    B --> C[LLM Agent]
    B --> D[Risk Manager]
    B --> E[Technical Indicator Engine]
    C --> F[LLM Provider (OpenAI / Gemini)]
    D --> G[Portfolio State (SQLite DB)]
    E --> H[Indicator Plug‑ins]
    G --> I[Order Execution Loop]
    I --> J[Trade Log & Equity Curve]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
```

---  

## Installation  

> **Prerequisites**  
> * **Python ≥ 3.10** (recommended 3.11)  
> * **Git** (for cloning)  
> * **Node ≥ 18** *(optional – only for building docs)*  

```bash
# 1️⃣ Clone the repository
git clone https://github.com/your-username/tradeiq.git
cd tradeiq

# 2️⃣ Create a virtual environment (highly recommended)
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS / Linux

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Optional: install development extras (pre‑commit, black, flake8)
pip install -r dev-requirements.txt
pre-commit install
```

---  

## Configuration  

All configurable values reside in `config.yaml`. Copy the template and edit as needed:

```bash
cp config.example.yaml config.yaml
```

Key sections:

| Section | Variable | Description |
|---------|----------|-------------|
| `llm` | `provider` | `"openai"` or `"gemini"` |
| | `api_key` | Your provider API key (keep secret) |
| `risk` | `max_position_pct` | Max portfolio exposure per asset |
| | `stop_loss_pct` | Fixed stop‑loss threshold |
| `database` | `path` | Path to SQLite DB (default `data/trading.db`) |
| `ui` | `theme` | `"dark"` / `"light"` (Streamlit theme) |

---  

## Running the Application  

```bash
streamlit run app.py
```

The UI will be served at `http://localhost:8501`.  
- **Dashboard** – Overview of portfolio equity, trade log, and live charts.  
- **Strategy Builder** – Select indicators, set LLM prompt templates, and tweak risk parameters.  

---  

## Demo & Screenshots  

| Screenshot | Description |
|-----------|-------------|
| ![Dashboard](file:///C:/Users/poora/.gemini/antigravity/artifacts/dashboard.png) | Main trading dashboard with equity curve and live order book. |
| ![Strategy Panel](file:///C:/Users/poora/.gemini/antigravity/artifacts/strategy_panel.png) | Strategy configuration panel (indicator selection, LLM prompt). |
| ![Risk Heatmap](file:///C:/Users/poora/.gemini/antigravity/artifacts/risk_heatmap.png) | Portfolio‑level VaR heatmap generated by the risk manager. |

*(Screenshots are placeholders – replace with actual images after a local run.)*  

---  

## Testing  

The project includes a comprehensive test suite powered by **pytest**.

```bash
# Run all unit and integration tests
pytest -v
```

Coverage report:

```bash
coverage run -m pytest
coverage html   # opens htmlcov/index.html
```

Key test modules:

* `tests/test_llm_agent.py` – verifies correct prompt formatting and mock LLM responses.  
* `tests/test_risk_manager.py` – validates stop‑loss, position‑size, and VaR calculations.  
* `tests/test_indicator_engine.py` – checks SMA/EMA, RSI, MACD outputs against known vectors.  

---  

## Contribution Guide  

We welcome contributions! Follow these steps:

1. **Fork** the repository.  
2. **Create a branch** for your feature or bug‑fix (`git checkout -b feature‑awesome‑logic`).  
3. **Write tests** for any new functionality.  
4. **Run linting & formatting** (`pre-commit run --all-files`).  
5. **Submit a Pull Request** with a clear description and reference any related issue.  

### Code Style  

* **Black** (line length 100) – automatic formatting.  
* **Flake8** – linting (ignore `E203`, `W503`).  
* **Docstrings** – Google style, placed at module, class, and function level.  

---  

## Roadmap  

| Milestone | Target Date | Description |
|-----------|-------------|-------------|
| **v1.0 – Public Release** | 2026‑06‑15 | Stable paper‑trading engine, full documentation, CI/CD pipeline. |
| **v1.1 – Broker Integration** | 2026‑08‑01 | Plug‑in for live trading with Alpaca / Interactive Brokers. |
| **v2.0 – Multi‑Agent Collaboration** | 2027‑01‑01 | Support multiple LLM agents negotiating on a single portfolio. |
| **v2.1 – Cloud Deployment** | 2027‑03‑15 | Docker + Kubernetes manifests, auto‑scaling on GCP/AWS. |

---  

## Citing the Project  

If you use TradeIQ in academic work or publications, please cite:

```bibtex
@software{tradeiq2026,
  author       = {Poora, Your Name},
  title        = {TradeIQ: An LLM‑Powered Paper Trading Portal},
  year         = 2026,
  month        = may,
  url          = {https://github.com/your-username/tradeiq},
  version      = {v1.0}
}
```

---  

## License  

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---  

## Acknowledgements  

* **OpenAI / Google Gemini** – for providing LLM APIs.  
* **Streamlit** – for the rapid UI prototyping framework.  
* **Hyperliquid‑trading‑agent** – inspiration for the agent‑risk‑indicator architecture.  
* **VADER Sentiment** – for sentiment extraction from news headlines.  

---  

### Happy Trading! 🚀  

Feel free to open an issue, start a discussion, or contribute a pull request. Your feedback helps make TradeIQ better for everyone.  
