# Polynomial Regression Example using MLflow projects

We’ve made a small project about polynomial regression models, which we will use to show how MLflow projects work. 

As in the previous exercises, click <> Code and open a codespace on main. In the terminal:

```bash
pip install mlflow
```

## Code explanation

The main experiment is defined in `experiment.py`. We simulate some data and model it using different degrees of polynomial regression: 

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import numpy as np

import sys
num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100

def get_ys(xs)
  signal = -0.1*xs**3 + xs**2 - 5*xs - 5
  noise = np.random.normal(0,100,(len(xs),1))
  return signal + noise

X = np.random.uniform(-20,20,num_samples).reshape((num_samples,1))
y = get_ys(X)

plt.scatter(X,y,label="data")

for degree in range(1,4):
    model = Pipeline([
      ("Poly", PolynomialFeatures(degree=degree)),
      ("LenReg", LinearRegression())
      ])
    model.fit(X,y)
    plotting_x = np.linspace(-20,20,num=50).reshape((50,1))
    preds = model.predict(plotting_x)
    plt.plot(plotting_x, preds, label=f"degree={degree}")

plt.legend()
plt.show()
```

The code above generates 100 (x,y) pairs by default (or the number of samples specified as a parameter by the user). The x values are chosen uniformly at random between (-20,20)

```python
X = np.random.uniform(-20,20,num_samples).reshape((num_samples,1))
```

and the y's are calculated to be a polynomial function of x 

```python
signal = -0.1*xs**3 + xs**2 - 5*xs - 5
```

with added noise 

```python
noise = np.random.normal(0,100,(len(xs),1))
```

We fit 3 polynomial models with degrees from 1, 2, and 3 to the data:

```python
for degree in range(1,4):
    model = Pipeline([
      ("Poly", PolynomialFeatures(degree=degree)),
      ("LenReg", LinearRegression())
      ])
    model.fit(X,y)
```

Suppose we want to share this code with someone else who will either simply run it or build on top of it. MLflow Projects come in handy.

## MLProjects

[MLflow Projects](https://mlflow.org/docs/latest/ml/projects/) provide a standard format for packaging and sharing reproducible data science code. Based on simple conventions, MLProjects enable seamless collaboration and automated execution across different environments and platforms.

Every MLflow Project consists of three key elements:

### Project name

A human-readable identifier for your project, typically defined in the `MLproject` file (have a look at it). Here, the name of the project is `name: PolyReg`.

### Environment

The software environment contains all dependencies needed to run the project. MLflow supports multiple environment types. Here we used a pip environment specified using a `python_env.yaml` file:

```bash
# Dependencies required to build packages. This field is optional.
build_dependencies:
  - pip
# Dependencies required to run the project.
dependencies:
  - scipy
  - scikit-learn>0.23
  - numpy>1.19
  - matplotlib>3
```

### Entry Points

The MLproject file specifies commands that can be executed within the project. Entry points define:

- Parameters - Inputs with types and default values
- Commands - What gets executed when the entry point runs
- Environment - The execution context and dependencies

In our case the MLproject file specifies that we are using the python environment `python_env.yaml` and the command to run the code is `python experiment.py {num_samples}`, where the parameter `num_samples` is of type `int` and which, if not specified, will take the value 100 by default.

```bash
name: PolyReg

python_env: python_env.yaml
# or
# conda_env: my_env.yaml
# or
# docker_env:
#    image:  mlflow-docker-example

entry_points:
  main:
    parameters:
      num_samples: {type: int, default: 100}
    command: "python experiment.py {num_samples}"
```

## Running the project

With these three things in place, you can run the experiment using `mlflow run <path to project>`. So, if you are located in the project directory, run:

```bash
mlflow run .  --env-manager=local
```

Once you've done this you might notice a new file `plot.png`; open it. It shows the generated (x,y) points and the 3 polynomial models. Which model is the best fit for the data and why?

Because this project is hosted as a git repository, you can simply do:

```bash
mlflow run https://github.com/LSDA-BDM/exercise-polynomial.git  --env-manager=local
```
This will fetch the project, resolve the environment, and run the main entry point with the default parameters.
If you have a simple environment with mlflow installed on your laptop, you can try running the above command on your machine too.

Let's say you want to run the experiment with 500 samples, instead of the default 100. Then you can do:

```bash
mlflow run https://github.com/LSDA-BDM/exercise-polynomial.git -P num_samples=500  --env-manager=local
```
