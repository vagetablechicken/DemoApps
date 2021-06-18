#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 4Paradigm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import composeml as cp
import utils

spark = SparkSession.builder.appName("Dataframe demo").getOrCreate()

train_data = utils.load_data('data/train_FD004.txt')

def remaining_useful_life(df):
    return len(df) - 1
lm = cp.LabelMaker(
    target_entity='engine_no',
    time_index='record_time',
    labeling_function=remaining_useful_life,
)
label_times = lm.search(
    train_data.sort_values('record_time'),
    num_examples_per_instance=1,
    minimum_data=100,
    verbose=True,
)
label_times.set_index('engine_no', inplace=True)

# filter train_data
need_drop = []
for idx, row in train_data.iterrows():
    if row['record_time'] > label_times.loc[row['engine_no'],'time']:
        need_drop.append(idx)

filtered = train_data.drop(need_drop)
filtered.sort_values(by='engine_no')

# spark calc
print('spark:')
train_df = spark.createDataFrame(filtered)
train_df.createOrReplaceTempView("t1")

sql_tpl = """
select
engine_no,
max(operational_setting_1) over w1,
max(operational_setting_2) over w1,
max(operational_setting_3) over w1,
max(sensor_measurement_1) over w1,
max(sensor_measurement_10) over w1,
max(sensor_measurement_11) over w1,
max(sensor_measurement_12) over w1,
max(sensor_measurement_13) over w1,
max(sensor_measurement_14) over w1,
max(sensor_measurement_15) over w1,
max(sensor_measurement_16) over w1,
max(sensor_measurement_17) over w1,
max(sensor_measurement_18) over w1,
max(sensor_measurement_19) over w1,
max(sensor_measurement_2) over w1,
max(sensor_measurement_20) over w1,
max(sensor_measurement_21) over w1,
max(sensor_measurement_3) over w1,
max(sensor_measurement_4) over w1,
max(sensor_measurement_5) over w1,
max(sensor_measurement_6) over w1,
max(sensor_measurement_7) over w1,
max(sensor_measurement_8) over w1,
max(sensor_measurement_9) over w1,
max(operational_setting_1) over w2,
max(operational_setting_2) over w2,
max(operational_setting_3) over w2,
max(sensor_measurement_1) over w2,
max(sensor_measurement_10) over w2,
max(sensor_measurement_11) over w2,
max(sensor_measurement_12) over w2,
max(sensor_measurement_13) over w2,
max(sensor_measurement_14) over w2,
max(sensor_measurement_15) over w2,
max(sensor_measurement_16) over w2,
max(sensor_measurement_17) over w2,
max(sensor_measurement_18) over w2,
max(sensor_measurement_19) over w2,
max(sensor_measurement_2) over w2,
max(sensor_measurement_20) over w2,
max(sensor_measurement_21) over w2,
max(sensor_measurement_3) over w2,
max(sensor_measurement_4) over w2,
max(sensor_measurement_5) over w2,
max(sensor_measurement_6) over w2,
max(sensor_measurement_7) over w2,
max(sensor_measurement_8) over w2,
max(sensor_measurement_9) over w2
from {}
window w1 as (partition by engine_no order by record_time rows_range between unbounded preceding and current row),
w2 as (partition by time_in_cycles order by record_time rows_range between unbounded preceding and current row)
;
"""

fm_sql = sql_tpl.format('t1')
fm_res = spark.sql(fm_sql)
fm = fm_res.toPandas()

fm = fm.groupby(['engine_no']).last().reset_index()
# print(fm)
# add remaining_useful_life col
fm = fm.set_index('engine_no').join(label_times[['remaining_useful_life']])

fm.to_csv('train_fm.csv')
print("save the feature matrix done")