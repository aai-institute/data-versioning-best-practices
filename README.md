# Data Version Control Best Practices Tutorial

Welcome 👋 

This contains a tutorial in the `data_versioning_best_practices.ipynb` to guide you through the essential practices for effectively versioning data in a data science project. 

We will be using lakeFS and appliedAI's filesystem wrapper lakeFS-spec to facilitate the versioning operations. 

Before you jump in, there are some setup-steps: 

1. Environment Setup
 Create a local virtual environment and install dependencies:

```bash
python -m venv .venv
```
Then, activate the virtual environment and Install all required dependencies by running:

```bash
pip install -r requirements.txt
```
2. Docker Installation and Setup
To run a local lakeFS backend, we use docker. If you don't have it installed follow the instructions [here](https://docs.docker.com/get-docker/).

Then in the repository root run 

```bash 
docker-compose up
```

This will start a lakeFS instance on port 8000.

3. Jupyter Notebook for Demonstration

Now you can open the `data_versioning_best_practices.ipynb` jupyter notebook, read through it and follow the steps. 

Happy Learning!