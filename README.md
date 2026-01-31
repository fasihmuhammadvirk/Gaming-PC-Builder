# Gaming PC Builder

## Overview
**Gaming PC Builder** is a Python-based application that helps users build their dream PC based on a specific budget and use case (e.g., Gaming, Workstation, Server). It utilizes a constraint satisfaction algorithm to select optimal components (CPU, GPU, RAM, Motherboard, Storage, PSU) from a dataset of real-world parts.

The application currently features a web interface built with [Streamlit](https://streamlit.io/).

## Features
- **Budget-Based Allocation**: Automatically optimizes component selection to fit within a user-defined budget.
- **Use Case Presets**: Pre-defined configurations for different needs:
  - General
  - Gaming
  - Workstation
  - Server
  - Custom
- **Constraint Checking**: Ensures compatibility between components (e.g., CPU socket matches Motherboard, PSU wattage is sufficient).
- **Interactive UI**: Adjust preferences and see results instantly.

## Project Structure
```
Gaming PC Builder/
├── Allocator/                 # Source code for the application
│   ├── allocator.py           # Core logic for component selection
│   ├── streamlit_app.py       # Streamlit frontend
│   └── requirements.txt       # Python dependencies
├── Datasheets/                # CSV Data files for components
│   ├── CPU/
│   ├── GPU/
│   ├── Motherboard/
│   ├── RAM/
│   ├── PSU/
│   └── Storage/
├── .gitignore
├── requirements.txt           # Project-level dependencies
└── README.md                  # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher

### Setup
1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository_url>
   cd "Gaming PC Builder"
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command from the root directory:

```bash
streamlit run Allocator/streamlit_app.py
```

This will launch the application in your default web browser.

## Customization
- **Data Updates**: You can update the component data by adding or modifying CSV files in the `Datasheets` directory.
- **Logic**: Modify `Allocator/allocator.py` to change the selection algorithm or constraints.
- **UI**: Modify `Allocator/streamlit_app.py` to change the web interface.

## License
[MIT](LICENSE)
