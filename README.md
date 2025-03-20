# Scientific Computing (Group 1)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation
To run this project locally, follow the steps below (replace [n] with the set number you want to access and run):

1. Clone the repository:
   ```bash
   git clone https://github.com/shaniasinha/sc-set3.git
   cd sc-set3
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook:
   ```bash
   jupyter notebook set3.ipynb
   ```

## Usage
- Open `set3.ipynb` in Jupyter Notebook for problem set 3.
- Follow the steps outlined in the notebook to run simulations and re-create the results.

## Project Structure
```
set_3/
│-- set3.ipynb                            # Jupyter notebook to run all code related to problem set 3
│-- requirements.txt                      # Dependencies
│-- src/                                  # Contains all classes and functions imported in the notebook
   │-- matrix_form_plot.py [*]
   |-- eigenmodes.py
   |-- direct_steady_state.py
   |-- leapfrog.py
   |-- video.py
│-- results/                              # Plots and animations 
   |-- direct_steady_state/                         
   │-- leapfrog/
   |-- misc/ [*]                          
│-- misc/                                 # Contains the problem set handout file and git fame results
   │-- archive/                           # Earlier implementations that could be useful for later
```
[*] note: `note` is note.

## Contributing
<div align="left">
    <img src="misc/git-fame.png" alt="Git fame set 3" width="600"/>
</div>

## Authors
- Myriam Belkhatir
- Shania Sinha
- Salomé Poulain

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
