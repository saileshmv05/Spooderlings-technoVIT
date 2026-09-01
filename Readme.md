# StockDNA

**Multi-Agent Autonomous Financial Intelligence System for Retail Investors**

Built for **HACKVERSE: INTO THE WEB — Sprint 1 · 24-Hour Hackathon · VIT Chennai 2026** (Problem Statement PS-01)

A single-file Streamlit app running a 6-agent CrewAI crew that converts live NSE data, SEBI filings, news and macro signals into explainable, personalized investment intelligence for retail investors.

---

## 🧠 Crew Roster

| # | Agent | Role |
|---|-------|------|
| 1 | **Technical Analyst** | RSI-14 momentum, volume anomaly, 20-SMA trend |
| 2 | **Fundamental Analyst** | SEBI filing RAG with chunk attribution |
| 3 | **Risk Advocate** | Adversarial downside / debt / bubble hunt |
| 4 | **Compliance & Governance** | Pledging, auditor, governance red flags |
| 5 | **Macro Analyst** | Rates, India VIX, crude, global sentiment |
| 6 | **Synthesis Committee** | Fusion, conflict resolution, risk adaptation |

## ✨ Design Pillars

- **Explainable** — every agent output carries evidence + citations
- **Personalized** — identical inputs yield *different* verdicts per risk profile
- **Resilient** — degraded-data fallbacks lower confidence, never crash, never emit uncited output
- **Auditable** — full reasoning trace + one-click JSON export
- **Offline-safe** — runs on a deterministic simulated LLM (zero API keys required)
- **AI Consult** — optional Gemini-powered chat consultant

## 🚀 Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs fully offline out of the box (simulated LLM, no keys needed). To enable the optional AI Consultant chat, add a Gemini API key in the sidebar or set:

```bash
export GEMINI_API_KEY="your-key-here"
```

## 📊 Features

- **Live/simulated NSE data** across 7 sectors (Banking, IT, Metals, Auto, Energy, Pharma, FMCG)
- **Risk-profile personalization**: Aggressive / Balanced / Conservative
- **Sector Screener** — ranks every stock in a sector cluster
- **Historical Deep-Dive** — 1-year volatility, 52-week range, drawdown, SMA crossovers
- **AI Consultant** — chat with Gemini about the verdict, risks, and next steps
- **Devil's Advocate** — automatic bear/bull contradiction builder
- **Full audit trail** — downloadable JSON session logs

## 🏗️ Architecture

Six specialist agents run in a **fan-out → fan-in** topology: five specialists produce structured JSON outputs (`agent_id, signal, score, confidence, data_quality, summary, evidence[], citations[]`), and a synthesis committee fuses them into a single cited, confidence-scored verdict.

**Composite score:**
```
composite = Σ(profile_weight × score × confidence × data_quality) / Σ(profile_weight × confidence × data_quality)
adjusted  = composite − volatility_penalty + profile_tilt
```

**Verdict thresholds:** STRONG BUY / BUY / HOLD / AVOID, based on the adjusted composite score.

## 🛡️ Resilience

- Market feed timeout → simulated OHLCV fallback, confidence ×0.55, labeled `DEGRADED`
- SEBI filings missing → `data_quality=missing`, empty citations, near-zero confidence
- News/macro down → curated fallback feeds, confidence haircut
- Conflicting agents → surfaced as `CONFLICT` with tie-break rationale

## 📁 Tech Stack

- Streamlit + Plotly
- CrewAI (multi-agent orchestration)
- yfinance (optional live data)
- feedparser (optional news feed)
- Google Generative AI (optional, for AI Consultant)

---

*Educational information only — not personalized investment advice.*
