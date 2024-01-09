# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Data Versioning Best Practices

This notebook will guide you through the best practices for versioning data in a data science project.
We assume you have lakeFS and lakeFS-spec set up. For guidance on setup and configuration, check the lakeFS-spec documentation.

We will explain the following best practices for data versioning in this example:
- [Set-up data repository](#set-up-a-data-repository)
- [Follow a branching strategy that ensures data integrity on the `main` branch, e.g. by running tests on feature branch datasets.](#branching-strategy)
- [Utilize reusable and tested functions for data transformation.](#utilize-reusable-and-tested-functions-for-data-transformation)
- [Use commits to save checkpoints and merges for atomic changes.](#use-commits-as-checkpoints-and-merge-branches-for-atomic-changes)
- [Keep naming (of branches and commits) consistent, concise, and unique. Use descriptive naming where it matters.](#descriptive-tags-for-human-readability-and-unique-SHAs-for-identification)

For this demo project, we aim to build a weather predictor using data from a public API.
This simulates a world scenario where we continuously collect more data, albeit with less complexity.
"""

# %%
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import lakefs
import pandas as pd
import sklearn
import sklearn.model_selection

import lakefs_spec

# %% [markdown]
"""
The cell below contains a helper function to obtain the data and code to obtain the default lakeFS credentials. It is otherwise not relevant to this demonstration.
"""


# %%
def _maybe_urlretrieve(url: str, filename: str) -> str:
    # Avoid API rate limit errors by downloading to a fixed local location
    destination = Path(tempfile.gettempdir()) / \
        "lakefs-spec-tutorials" / filename
    destination.parent.mkdir(exist_ok=True, parents=True)
    if destination.exists():
        return str(destination)

    outfile, _ = urllib.request.urlretrieve(url, str(destination))
    return outfile


outfile = _maybe_urlretrieve(
    "https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2010-01-01&end_date=2010-12-31&hourly=temperature_2m,relativehumidity_2m,rain,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m",
    "weather-2010.json",
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/aai-institute/lakefs-spec/main/docs/tutorials/.lakectl.yaml",
    os.path.expanduser("~/.lakectl.yaml"),
)
# %% [markdown]
"""
## Set up a data repository

We got the data for the year 2010. That should be enough for initial prototyping.
Later, however, we want to use more data. Since our dataset will be evolving, we implement data version control.
This ensures the reproducibility of our experiments, enables collaboration with colleagues, and allows our dynamic dataset to stay a valuable asset over time.

To set up versioning, we need to decide on a versioning tool (lakeFS in our case), set up a repository, and define which data is considered in scope and should be versioned and which is not.

We will interface with the lakeFS server using lakeFS-spec, our filesystem implementation for lakeFS.
"""

# %%
REPO_NAME = "weatherpred"

fs = lakefs_spec.LakeFSFileSystem()
repo = lakefs.Repository(REPO_NAME, fs.client).create(
    storage_namespace=f"local://{REPO_NAME}")


# %% [markdown]
"""
lakeFS works similarly to the Git versioning system and shares many of its core concepts.
You can create *commits* that contain specific changes to the data.
This commit captures an immutable state of the data at the time of the commit, rather than just saving a delta. This way, you do not have to reiterate over all previous commits to recoup the state of the repository at the commit. 
You can also work with *branches* to create an isolated view of the data.
Every commit (on any branch) is identified by a unique commit ID, a unique identifier obtained via a hashing function, also called SHA.

## Branching Strategy

We recommend following a branching strategy that ensures the data integrity on the main branch.
Since we are about to do some data wrangling, we will branch off the default branch `main` and later merge back into it, once we are sure everything works as expected.
"""

# %%
BRANCH = "transform-raw-data"

with fs.transaction as tx:
    fs.put(outfile, f"{REPO_NAME}/{BRANCH}/weather-2010.json")
    tx.commit(repository=REPO_NAME, branch=BRANCH,
              message="Add 2010 weather data")

# %% [markdown]
"""
## Utilize reusable and tested functions for data transformation

Now that we have the data on the `transform-raw-data` branch, we can start with the transformation.

It is good practice to encapsulate common transformations in (composable) functions. We use type
"""


# %%
def load_json_data(filepath: str | Path) -> dict:
    """Load JSON data from a file or file-like object."""
    if isinstance(filepath, Path):
        filepath = str(filepath)
    with open(filepath, "r") as f:
        return json.load(f)


def create_dataframe_from_json(json_data: dict) -> pd.DataFrame:
    """Create a Pandas DataFrame from a JSON object's "hourly" key."""
    df = pd.DataFrame.from_dict(json_data["hourly"])
    return df


def convert_time_column_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime format of DataFrame 'time' column."""
    df["time"] = pd.to_datetime(df["time"])
    return df


def add_rain_indicators(df: pd.DataFrame) -> pd.DataFrame:
    "Add dummy indicator variables for current and 24-hour future rain to each row based on timestamp and `rain` column. Remove columns with not missing values (i.e. end of DataFrame)."
    df["is_raining"] = df.rain > 0
    df["is_raining_in_1_day"] = df.is_raining.shift(24).astype(bool)
    return df.dropna()


# %% [markdown]
"""
We also add unit tests for our data transformation functions. This ensures the accuracy and reliability of data processing.
We catch errors early make our code maintainable and scalable code as we ensure that changes or additions to the code don't break existing functionalities.
"""

# %%
sample_json_data = {
    "hourly": [{"time": "2023-01-01 00:00", "rain": 0}, {"time": "2023-01-01 01:00", "rain": 2}]
}

df_test = create_dataframe_from_json(sample_json_data)
assert isinstance(df_test, pd.DataFrame), "Output should be a pandas DataFrame"
assert list(df_test.columns) == [
    "time", "rain"], "DataFrame should have time and rain columns"
assert len(df_test) == 2, "DataFrame should have two rows"


df_test = pd.DataFrame(
    {"time": ["2023-01-01 00:00", "2023-01-01 01:00"], "rain": [0, 2]})

df_test = convert_time_column_to_datetime(df_test)
assert pd.api.types.is_datetime64_any_dtype(
    df_test["time"]), "Time column should be of datetime type"


df_test = pd.DataFrame(
    {"time": pd.date_range(start="2023-01-01", periods=48,
                           freq="H"), "rain": [0] * 24 + [2] * 24}
)

df_test = add_rain_indicators(df_test)
assert ("is_raining" in df_test.columns and "is_raining_in_1_day" in df_test.columns), "Both indicator columns should be present"
assert all(df_test.loc[24:, "is_raining"]
           ), "All values should be True in 'is_raining' for the second day"
assert all(df_test.loc[:23, "is_raining_in_1_day"]
           ), "All values should be True in 'is_raining_in_1_day' for the first day"
assert not df_test.isna().any(
    axis=None), "There should be no NaN values in the DataFrame"

# %% [markdown]
"""
Encapsulating data processing steps in unit tested functions (which are also made available to the whole team or organisation) saves some work by reusing our code.
Additionally, the tests increase our confidence in the data quality and serve as additional context to infer the purpose of the function should we or someone else come back at a later time.
Typehints, docstrings, and unit tests serve as documentation and help our peers (and ourself revisiting in six months time) to easily understand the code.

We can now apply the functions to process the data.
"""

# %%
json_data = load_json_data(outfile)
df = create_dataframe_from_json(json_data)
df = convert_time_column_to_datetime(df)
df = add_rain_indicators(df)

df.head(5)

# %% [markdown]
"""
## Use commits as checkpoints and merge branches for atomic changes

Now we can commit the updated data to the `transform-raw-data` branch in the lakeFS repository.
We write a descriptive commit message.
"""
# %%
with fs.transaction as tx:
    df.to_csv(f"lakefs://{REPO_NAME}/{BRANCH}/weather_2010.csv")
    commit = tx.commit(repository=REPO_NAME, branch=BRANCH,
                       message="Preprocess 2010 data")
print(commit)

# %% [markdown]
"""
We see that the commit has a unique id, the commit SHA, which we can log to identify this particular state of the data.

We are done with our processing steps. However, before we merge the branch back into `main`, we should follow a review process.
How this review process looks heavily depends on the project domain and maturity as well as the nature of the data itself.
It can also consist of manual and/or automated checks. Manual checks are much like code reviews.
Automated checks are more complicated to set up and there are few one size fits all solutions.
For example, checking for distribution drift does not make sense if you add data from a previously underrepresented class.

We still want to present some keywords and links that serve as a starting point for you:

- [Data Quality Testing blogpost by lakeFS](https://lakefs.io/data-quality/data-quality-testing/)
- [ArXiv paper on distribution drift measurement](https://arxiv.org/abs/1908.04240)
- [Dynamic vs. static data testing by Anomalo](https://www.anomalo.com/post/dynamic-data-testing?gi=fb4db0e2ecb4)


Nonetheless, there are probably some basic, heuristic data quality checks, like ensuring no NaNs, that are easy to implement and likely catch big processing erros.
"""

# %%
assert not df.isna().any(axis=None), "There should be no NaN values in the DataFrame"
assert ((df[['is_raining', 'is_raining_in_1_day']].isin([0, 1])).all(
    axis=None)), "Values in 'is_raining' and 'is_raining_in_1_day' should be only 0 or 1"

# %% [markdown]

"""
As our data looks good, we can go ahead and merge it.
"""
# %%
with fs.transaction as tx:
    tx.merge(repository=REPO_NAME, source_ref=BRANCH, into="main")

# %% [markdown]
"""
We will now start to develop our ML model. We recommend creating a new branch for each experiment for proper separation.
There, we will conduct the train test split and further experiment specific modifications, if applicable.
"""

# %%
EXPERIMENT_BRANCH = "experiment-1"
with fs.transaction as tx:
    tx.create_branch(repository=REPO_NAME,
                     name=EXPERIMENT_BRANCH, source="main")

df = pd.read_csv(f"lakefs://{REPO_NAME}/{EXPERIMENT_BRANCH}/weather_2010.csv")
model_data = df.drop("time", axis=1)
train, test = sklearn.model_selection.train_test_split(
    model_data, random_state=7)

# %% [markdown]
"""
## Descriptive tags for human readability and unique SHAs for identification

Since the train/test split data set we just committed to the experiment branch is something we expect to address quite often in development, we will also add a human-readable tag to this commit.
In the code below we directly pass the commit as the reference. If we pass a branch, the tag will point to the current HEAD (i.e., the latest) commit on the branch.
This makes it easy to refer back to it and to communicate this specific state of the data to colleagues. Tags are immutable which ensures consistency.
You and your colleagues can then also work with the same state (i.e., train/test split, etc.) of the data by referring to the tag by name.

"""

# %%
TAG_NAME = "exp1_2010_data"
with fs.transaction as tx:
    train.to_csv(f"lakefs://{REPO_NAME}/{EXPERIMENT_BRANCH}/train_weather.csv")
    test.to_csv(f"lakefs://{REPO_NAME}/{EXPERIMENT_BRANCH}/test_weather.csv")
    commit = tx.commit(
        repository=REPO_NAME,
        branch=EXPERIMENT_BRANCH,
        message="Perform train-test split of 2010 weather data",
    )
    tx.tag(repository=REPO_NAME, ref=commit, tag=TAG_NAME)

# %% [markdown]
"""
Now we have the data on different branches. If new data comes in, we can perform necessary preprocessing on a separate branch and merge it to `main` once we are sure about its compatibility and have run all necessary tests.

Should the new data be important for the experimentation as well, then we can merge the updated `main` branch into the experimentation branch.
If we expect this version to be referred to by humans often, we can create a new tag for the dataset.

It is important to note that tags cannot be directly reassigned. This is to ensure reproducibility of previously written code.

If you want to reuse tags anyways, for example if you have too many stale tags assigned, you have to delete the tag and create a new one.
However, be aware as this might break reproducibility in other places (i.e., colleagues might expect unchanged data).
To ensure failsafe versioning, use commit IDs instead of tags or branch names in logs or experiment tracking tools.
"""
