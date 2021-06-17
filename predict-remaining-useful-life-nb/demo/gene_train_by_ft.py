import composeml as cp
import numpy as np
import pandas as pd
import featuretools as ft
import utils
import os

data_path = 'data/train_FD004.txt'
data = utils.load_data(data_path)
def remaining_useful_life(df):
    return len(df) - 1
lm = cp.LabelMaker(
    target_entity='engine_no',
    time_index='time',
    labeling_function=remaining_useful_life,
)
label_times = lm.search(
    data.sort_values('time'),
    num_examples_per_instance=1,
    minimum_data=100,
    verbose=True,
)
def make_entityset(data):
    es = ft.EntitySet('Dataset')

    es.entity_from_dataframe(
        dataframe=data,
        entity_id='recordings',
        index='index',
        time_index='time',
    )

    es.normalize_entity(
        base_entity_id='recordings',
        new_entity_id='engines',
        index='engine_no',
    )

    es.normalize_entity(
        base_entity_id='recordings',
        new_entity_id='cycles',
        index='time_in_cycles',
    )

    return es
es = make_entityset(data)
fm, features = ft.dfs(
    entityset=es,
    target_entity='engines',
    agg_primitives=['max'],
    trans_primitives=[],
    cutoff_time=label_times,
    max_depth=3,
    verbose=True,
)

fm.to_csv('simple_fm.csv')

