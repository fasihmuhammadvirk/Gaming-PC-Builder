# �️ Intelligent PC Component Allocator

## Project Overview
This project serves as a practical implementation of **Constraint Satisfaction** and **Multi-Objective Optimization** systems applied to hardware selection. By leveraging real-world component datasets, the system automates the process of generating optimal PC configurations based on strict budget constraints and variable user utility functions (e.g., Gaming vs. Workstation).

Built with **Python**, **Pandas**, and **Streamlit**, this application transforms raw hardware specifications into actionable insights through a weighted scoring model.

## ⚙️ Algorithmic Architecture

### 1. Data Acquisition & Feature Engineering
One of the primary challenges addressed in this project was the construction of a high-fidelity dataset from unstructured web sources.
- **Automated Data Mining**: Designed custom extraction pipelines to scrape and aggregate disparate component specifications from multiple online vendors and technical databases.
- **Advanced Preprocessing (ETL)**: Transformed raw, noisy web data into a structured schema. This involved complex text parsing to extract numerical specifications (e.g., "3.5GHz" -> `3.5`), handling missing data via heuristic imputation, and normalizing inconsistent nomenclature across different manufacturers.
- **Derived Metrics**: Engineered synthetic features such as `Price-to-Performance` index and `Wattage-Per-Core` to drive the optimization logic.

### 2. The Optimization Algorithm
The core allocator (`allocator.py`) treats the PC building process as a specialized optimization problem:
- **Objective Function**: Maximize total `Component Score` based on component attributes (Clock Speed, VRAM, Cache, etc.).
- **Constraints**:
  - **Hard Constraints**: Compatibility (Socket matching, Total Wattage < PSU Capacity, RAM Generation support).
  - **Soft Constraints**: Budget allocation per component category.
- **Weighted Scoring Model**: Each use-case (Gaming, Server, Workstation) is defined as a specific **Weight Vector**. For example, the `Gaming` vector heavily weights Single-Core IPC and GPU Floating Point performance, whereas `Server` prioritizes Core Count and RAM Capacity.

```python
# Simplified Logic Representation
Score = Σ (Feature_i * Weight_i)
Constraint: Σ (Cost_j) <= Total_Budget
Constraint: Socket_CPU == Socket_Motherboard
```

### 3. Adaptive Budgeting
The system uses a dynamic allocation strategy. If a component (e.g., GPU) undercuts its allocated budget, the `leftover` capital is recursively re-distributed to subsequent components to maximize the overall system utility.

## 🛠 Tech Stack
- **Data Processing**: `pandas`, `numpy`
- **Frontend/Dashboarding**: `streamlit`
- **Backend Logic**: Python
- **Data persistence**: CSV (Flat file database)

## 📂 Repository Structure
```
Gaming PC Builder/
├── Allocator/
│   ├── allocator.py           # Optimization Engine & Constraint Logic
│   ├── streamlit_app.py       # Data Visualization & Comparison Dashboard
│   └── requirements.txt       # Dependencies
├── Datasheets/                # The Dataset (Cleaned Component Specs)
│   ├── CPU/                   # Normalized CPU specs
│   ├── GPU/                   # Normalized GPU specs
│   └── ...
├── run.sh                     # Automated Environment Setup & Execution Script
└── requirements.txt           # Project Dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Quick Start (Automated)
We provide a shell script to handle environment creation, dependency installation, and execution automatically:

```bash
chmod +x run.sh
./run.sh
```

### Manual Installation
1.  **Clone the repo**
    ```bash
    git clone <repo_url>
    cd "Gaming PC Builder"
    ```
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Dashboard**
    ```bash
    streamlit run Allocator/streamlit_app.py
    ```

## 📊 Dataset Attribution
The datasets used in this project are aggregated from various hardware benchmarks and retailer specifications, cleaned and normalized for algorithmic processing.

## License
[MIT](LICENSE)
