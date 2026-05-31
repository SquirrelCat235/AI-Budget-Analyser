# 💸 AI-Powered Budget Analyzer & Predictor

A personal finance dashboard built with Python and Streamlit that analyzes your spending patterns, predicts next month's expenses using machine learning, and delivers personalized financial recommendations.



## ✨ Features

- **Interactive Dashboard** — View total spending, top categories, and breakdowns for any selected month
- **Visual Analytics** — Donut chart for category breakdown and a daily spending line chart powered by Plotly
- **ML-Based Predictions** — Uses Linear Regression to forecast next month's total and per-category expenses based on historical trends
- **Smart Recommendations** — Auto-generated financial tips based on your actual spending behavior
- **CSV Upload Support** — Plug in your own data or explore with built-in dummy data


## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Web app framework |
| [Pandas](https://pandas.pydata.org/) | Data processing |
| [NumPy](https://numpy.org/) | Numerical computation |
| [Plotly](https://plotly.com/) | Interactive charts |
| [Scikit-learn](https://scikit-learn.org/) | Linear regression model |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-budget-analyser.git
cd ai-budget-analyser

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Dependencies

```
streamlit
pandas
numpy
plotly
scikit-learn
```

Or create a `requirements.txt` with the above and run `pip install -r requirements.txt`.


## 📂 CSV Format

To upload your own data, provide a CSV file with these three columns:

| Date | Category | Amount |
|---|---|---|
| 2025-01-05 | Groceries | 850.00 |
| 2025-01-05 | Dining Out | 450.00 |

Supported categories (customizable): `Groceries`, `Rent`, `Utilities`, `Entertainment`, `Dining Out`, `Transportation`, `Healthcare`, `Miscellaneous`


## 📊 How It Works

1. **Upload** a CSV or use the built-in demo data
2. **Select a month** from the sidebar to explore your spending
3. **View predictions** — the app trains a linear regression model on your monthly totals and forecasts the next month
4. **Read recommendations** — rule-based insights flag overspending, dining habits, trend changes, and category-level risk


## 🖼️ Demo

No CSV? No problem. The app auto-generates realistic dummy transaction data for the current year so you can explore all features right away.




