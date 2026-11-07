# 🤖 City Time & Weather Agent (`city_time_weather_agent`)

> **The Reference Implementation for the 13-Day AI Agent Evaluation & Quality Engineering Series**  
> *Closing the loop from `pytest` & `adk eval` to Ambient Quality Agent (AQuA) & LangGraph / LangSmith.*

![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)
![Google ADK](https://img.shields.io/badge/Google%20ADK-2.13%2B-teal.svg)
![Evaluation](https://img.shields.io/badge/Evaluation-6--Layer%20Flywheel-orange.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-Gate%200%20to%20Gate%205-green.svg)

---

## 📌 Overview

**`city_time_weather_agent`** is a dual-tool reasoning agent built on **Google Agent Development Kit (ADK)** and powered by Gemini. It resolves real-time meteorological conditions and accurate local timezones via Open-Meteo APIs, backed by deterministic static fallbacks.

While the agent itself is lightweight and reproducible, it serves as the **core benchmark** for a 13-day deep dive into **Enterprise AI Agent Evaluation**. Rather than treating evaluation as an afterthought or a single post-hoc prompt, this repository demonstrates how to architect a complete, multi-tiered **Quality Flywheel** from line 1 of code to 24/7 production observability.

---

## 🏛️ The 6-Layer AI Agent Quality Flywheel

Every technique, test file, and evaluation configuration in this repository maps directly to one of the 6 layers of the flywheel:

![Six-Layer AI Agent Quality Flywheel](Six-Layer%20AI%20Agent%20Quality%20Flywheel.png)

```
                  ┌──────────────────────────────────────────────┐
                  │ 🔄 CLOSED-LOOP AGENT QUALITY FLYWHEEL        │
                  │   Production Bugs Become Tomorrow's Tests    │
                  └──────────────────────┬───────────────────────┘
                                         │
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Phase 1: Foundations & Golden Data (Gate 0)                                   │
 │ • Deterministic Tool Unit Tests (<0.5s, $0 cost)                              │
 │ • OpenTelemetry Span Inspection (adk web)                                     │
 │ • Canonical EvalSet Schemas (4 Collection Methods)                           │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Phase 2: Deterministic CI Replay (Gate 1)                                     │
 │ • adk conformance record / test (<3s, $0 LLM token cost)                       │
 │ • AgentEvaluator in pytest                                                    │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Phase 3: Semantic Rubrics & Persona Simulation (Gate 2)                       │
 │ • LLM-as-a-Judge (Hallucination, Safety, Trajectory Quality)                  │
 │ • Multi-turn UserSimulator Persona Stress-Testing                             │
 │ • Custom Code Guardrail Metrics (Zero Math Integrity)                         │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Phase 4: Cloud Runtime & Release Promotion (Gate 3)                           │
 │ • Vertex AI Agent Engine & Gemini Enterprise Deployment                       │
 │ • Eventarc & Cloud Build Automated Gate                                       │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Phase 5: Online Drift & Ambient Quality Agent (AQuA) (Gates 4 & 5)            │
 │ • 24/7 Cloud Monitoring Alerts on Tool Quality Drift                          │
 │ • AQuA: BigQuery _AllSpans -> Self-Verification Critic -> Code-Anchored RCA   │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Phase 6: Cross-Framework Parity (Google ADK <-> LangGraph / LangSmith)         │
 │ • 1-to-1 Architectural Rosetta Stone & Portability Guide                      │
 └────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗓️ 13-Day LinkedIn Learning Series Roadmap

This repository evolves daily alongside the LinkedIn series. Follow along or jump directly to the specific gate you are implementing:

| Day | Status | Phase | Topic / Title | CI/CD Gate | Key Files Added / Highlighted |
| :-: | :---: | :--- | :--- | :--- | :--- |
| **0** | ✅ | Kickoff | **The 6-Layer AI Agent Quality Flywheel** | Overview | `Six-Layer AI Agent Quality Flywheel.png` |
| **1** | ✅ | Foundations | **Why Agent Evaluation $\neq$ Unit Testing** | **Gate 0** (Pre-commit / PR) | `tests/unit/test_tools_*.py`, `test_agent_contracts.py` |
| **2** | ✅ | Foundations | **Smoke-Testing & OpenTelemetry Span Inspection** | **Gate 0** (Dev Inner-Loop) | `agent.py`, `adk web` trace inspection |
| **3** | 📅 | Foundations | **The Golden Dataset: `EvalSet` & The 4 Collection Paths** | Git Artifact | `evalset1.evalset.json`, BigQuery trace miner |
| **4** | 📅 | Deterministic | **$0 Fast CI Replay with `adk conformance`** | **Gate 1** (Every Commit) | `tests/weather/generated-recordings.yaml` |
| **5** | 📅 | Deterministic | **Architecting Your `pytest` Suite (`AgentEvaluator`)** | **Gate 1 & 2** (PR Merge) | `tests/integration/test_agent_evaluator.py` |
| **6** | 📅 | Rubrics | **Offline LLM-as-a-Judge & Trajectory Metrics** | **Gate 2** (Staging Gate) | `eval_config.json`, `adk eval` |
| **7** | 📅 | Personas | **Dynamic Multi-Turn Persona Simulation** | **Gate 2** (Stress Gate) | `scenarios.json`, `persona_eval_config.json` |
| **8** | 📅 | Custom Code | **Domain-Specific Custom Code Metrics** | **Gate 2** (Compliance) | `custom_metrics.py` (fail-closed checks) |
| **9** | 📅 | Cloud Promotion | **Vertex AI Agent Engine & Cloud Build CI/CD Gate** | **Gate 3** (Release Promotion) | `deploy_adk.py`, `cloudbuild.yaml` |
| **10** | 📅 | Online Drift | **Continuous 24/7 Quality Drift Monitors & Alerts** | **Gate 4** (Continuous Prod) | `policy.yaml` (Cloud Monitoring) |
| **11** | 📅 | AQuA RCA | **AQuA: From Production Spans to Exact Code Fixes** | **Gate 5** (Nightly Audit) | `goal.md`, `scripts/run_aqua_eval.sh` |
| **12** | 📅 | Cross-Framework | **Rebuilding the Pipeline on LangGraph + LangSmith** | All Gates | `agent_langgraph.py`, `test_langgraph.py` |
| **13** | 📅 | Finale | **The Rosetta Stone Map: Google ADK $\leftrightarrow$ LangGraph** | Full Lifecycle | Complete 11-Layer Comparison Matrix |

*Legend: ✅ Completed | 🔄 In Progress / Next Up | 📅 Upcoming*

---

## 📂 Project Structure

```text
city_time_weather_agent/
├── agent.py                        # Core agent logic, tools & Open-Meteo fallback
├── requirements.txt                # Production & evaluation dependencies
├── pytest.ini                      # Pytest runner markers & pythonpath config
├── .gitignore                      # Security & cache exclusions (protects secrets & .env)
├── .env.example                    # Sample environment configuration
├── Dockerfile                      # Container definition for Cloud deployment
├── Six-Layer AI Agent Quality Flywheel.png # Hand-drawn master architecture diagram
└── tests/
    ├── conftest.py                 # Resilient sys.path injection, mock fixtures & cache resets
    ├── unit/                       # Layer 1A: Fast deterministic unit tests (Gate 0)
    │   ├── test_tools_weather.py   # Normalization, error boundaries, live API mocks, & geo cache
    │   ├── test_tools_time.py      # Timezone offset, strftime contracts, & fuzzy input handling
    │   └── test_agent_contracts.py # Tool docstrings, OpenAPI type hints, & prompt invariants
    └── integration/                # Layer 1B: Golden EvalSet & AgentEvaluator (Gate 1 & 2)
        └── test_agent_evaluator.py # (Added on Day 5)
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13
- A Google Cloud Project with Vertex AI enabled (for Days 3–11)

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone <your-repo-url>/city_time_weather_agent.git
cd city_time_weather_agent

python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install pytest pytest-asyncio
```

### 3. Environment Configuration
Copy the sample environment variables:

```bash
cp .env.example .env
```
Edit `.env` to configure your `GOOGLE_CLOUD_PROJECT` and location if running cloud/eval features.

---

## 🧪 Running the Evaluation Suites

### Gate 0: Fast Deterministic Tool Unit Tests (`<0.5s`, `$0.00` LLM Cost)
Before calling an LLM or recording traces, run the deterministic unit suite to verify tool contracts, input normalization, error boundaries, and schema docstrings:

```bash
pytest tests/unit/ -m unit
```

**What this verifies:**
1. **Schema & Docstrings**: Verifies tools have non-empty docstrings and strict type hints so Google ADK can generate valid OpenAPI function declarations.
2. **Graceful Error Boundaries**: Asserts that unknown cities (`Atlantis`) return `{"status": "error", ...}` instead of raising unhandled `KeyError` exceptions that crash the agent.
3. **Input Sanitization**: Tests whitespace trimming, uppercase strings, and localized country suffixes (`"Tokyo, Japan"`).
4. **State Isolation**: Resets the in-memory `_GEO_CACHE` between test runs.

---

### Interactive Smoke-Testing (Day 2 Preview)
To test the agent interactively via CLI:

```bash
adk run city_time_weather_agent
```

To inspect live OpenTelemetry execution spans (`call_llm` $\rightarrow$ `execute_tool` $\rightarrow$ `call_llm`):

```bash
adk web .
```

---

## 🛡️ Enterprise Failure Patterns & Fixes

Throughout this series, each post covers real-world failure patterns observed in enterprise deployments:

- **Missing Tool Docstrings**: Omitting docstrings in Python tool functions results in blank OpenAPI descriptions, causing the LLM to hallucinate or misroute function arguments. *(Addressed in Day 1)*
- **Crashing on Unhandled Exceptions**: Tools must never bubble up raw exceptions (`KeyError`, `URLError`) to the orchestrator. They must return structured error dictionaries that allow the LLM to apologize and self-correct. *(Addressed in Day 1)*
- **Replay Divergence on Unpinned Prompts**: Modifying prompt instructions invalidates deterministic recording hashes in `adk conformance`. *(Addressed in Day 4)*
- **Silent Distribution Shift**: Production users ask questions in formats absent from the training/evaluation dataset, triggering online monitor alerts. *(Addressed in Day 10)*
- **Unverified LLM-as-a-Judge Hallucinations**: Standard LLM judges produce false positives; AQuA's Self-Verification Critic filters candidate defects against raw session transcripts. *(Addressed in Day 11)*

---

## 🤝 Contributing & Updates

This repository is maintained as a public learning resource. If you encounter an edge case or have suggestions for new evaluation rubrics or CI/CD integrations:
1. Open an Issue with the tag `[Series Feedback]`.
2. Submit a Pull Request targeting the active day's branch.

---

## 📄 License
Apache 2.0. Feel free to use this architecture and test pattern in your own agentic applications!

