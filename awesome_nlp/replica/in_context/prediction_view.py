# To run use `streamlit run prediction_view.py`
import glob
import importlib
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st


def get_task_meta(name):
    module = importlib.import_module(f"dataset_readers.datasets.{name}")

    return module.DatasetWrapper


def main():
    cd = Path(__file__)

    sys.path.append(str(cd.parent))

    in_context_dir = cd.parents[3].joinpath("experiments", "in_context").as_posix()

    st.set_page_config(layout="wide")

    files = sorted(glob.glob(in_context_dir + "/**/preds-*.json", recursive=True), key=os.path.getmtime)

    m_t_l_run = {}

    for f in files:
        splits = Path(f).parts[-3:]
        method, task, run = splits

        m_t_l_run.setdefault(method, {})
        m_t_l_run[method].setdefault(task, [])
        m_t_l_run[method][task].append(run)

    col_method, col_task, col_run = st.columns([1, 1, 1])
    with col_method:
        method = st.selectbox("Methods", options=m_t_l_run.keys())
    with col_task:
        task = st.selectbox("Tasks", options=m_t_l_run[method].keys())
    with col_run:
        run = st.selectbox("Runs", options=m_t_l_run[method][task])

    @st.cache_resource
    def load_json(file):
        with open(file) as f:
            data = json.load(f)
        return data

    filename = os.path.join(in_context_dir, method, task, run)
    data = load_json(filename)
    task_meta = get_task_meta(task)

    num_in = st.sidebar.number_input("Pick a question", 0, len(data), key="num_in")
    curr_el = data[num_in]

    st.write(f"## Question")
    st.write(task_meta.fields["gen"](curr_el))

    labels_col, preds_col, acc_col = st.columns([1, 1, 1])

    labels = curr_el["labels"]
    predictions = curr_el["predictions"]

    with labels_col:
        st.write(f"## Labels")
        st.write(labels)
    with preds_col:
        st.write(f"## Predictions")
        st.write(predictions)
    with acc_col:
        st.write(f"## Results")
        st.write("✅" if labels[-2] == predictions[-2] else "❌")

    st.write(f"## Prompt")
    st.write(f"```\n" + curr_el["prompt"])

    st.write(f"## Probability")
    st.write(curr_el["logits"])


if __name__ == "__main__":
    main()
