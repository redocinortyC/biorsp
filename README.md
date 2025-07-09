# BioRSP

### How to test BioRSP

1. Create a virtual environment. You can use `conda` or `venv`. Here is how to do it with `venv`:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install the package in editable mode. This allows you to make changes to the code and have them reflected without reinstalling the package:

   ```bash
   python3 -m pip install -e .
   ```

3. Go to `examples` directory. Currently, there is a single example Jupyter notebook that demonstrates how to use the package.

Good luck!
