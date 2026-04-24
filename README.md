# Rossmann Staff Optimization

A prescriptive optimization project implementing Mixed Integer Programming (MIP) and Genetic Algorithm (GA) approaches for optimizing staff scheduling at Rossmann stores.

## Overview

This project develops and compares two optimization methodologies:
- **Prescriptive MIP Implementation**: Exact optimization using Mixed Integer Programming
- **Genetic Algorithm (GA) Implementation**: Heuristic-based optimization approach

## Features

- Staff shift scheduling optimization
- Multiple constraint handling
- Performance comparison between exact and heuristic methods
- Scalable to multiple store locations

## Project Structure

```
Rossmann-Staff-Optimisation/
├── README.md
├── requirements.txt
├── .gitignore
├── mip/
│   └── # Mixed Integer Programming implementations
├── ga/
│   └── # Genetic Algorithm implementations
├── data/
│   └── # Sample data and datasets
└── results/
    └── # Output and analysis results
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/amipanes/Rossmann-Staff-Optimisation.git
cd Rossmann-Staff-Optimisation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running MIP Optimization
```bash
python -m mip.main
```

### Running Genetic Algorithm
```bash
python -m ga.main
```

## Results

Comparative analysis and results are stored in the `results/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

**amipanes**

---

For questions or issues, please open an issue on GitHub.
