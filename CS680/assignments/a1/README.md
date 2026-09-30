# CS680 Assignment 1

This directory contains the Jupyter notebook for Assignment 1:

- `a1.ipynb`

## Requirements

This notebook was developed using:

- **Anaconda3**
- **Python 3.12.4**
- **Jupyter Notebook / JupyterLab**

Required Python packages:

- `numpy`
- `pandas`
- `matplotlib`

These packages are typically included with Anaconda3.

## Project Structure

```text
a1/
├── a1.ipynb
├── datasets/
├── utils/
└── README.md
```

The notebook expects datasets to be available in the local `datasets/` directory.

## How to Run

From a terminal with Anaconda enabled, navigate to this assignment directory:

```bash
cd ~/a1
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

or JupyterLab:

```bash
jupyter lab
```

Then open:

```text
a1.ipynb
```

Run the notebook cells in order.

## Notes

If you get a `FileNotFoundError` when loading files from `./datasets/`, make sure the notebook is running from the assignment directory:

```text
~/a1
```

You can check the current working directory inside the notebook with:

```python
import os
print(os.getcwd())
```

If needed, set it manually:

```python
import os
os.chdir("/Users/carlospineda/PycharmProjects/uw-f26/CS680/assignments/a1")
```
