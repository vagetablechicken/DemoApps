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

"""

"""

import numpy as np
import tornado.web
import tornado.ioloop
import json
import sqlalchemy as db
from sqlalchemy_fedb.fedbapi import Type as feType

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import utils

fm = pd.read_csv('train_fm.csv', index_col='engine_no')
X = fm.copy().fillna(0)
y = X.pop('remaining_useful_life')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)

# skip baselines

reg = RandomForestRegressor(n_estimators=100)
reg.fit(X_train, y_train)

preds = reg.predict(X_test)
scores = mean_absolute_error(preds, y_test)
print('[Train] Mean Abs Error: {:.2f}'.format(scores))


engine = db.create_engine('fedb:///db_test?zk=127.0.0.1:2181&zkPath=/fedb')
connection = engine.connect()

sql = """
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
""".format('rul_table')

TypeDict = {feType.Bool:"bool", feType.Int16:"smallint", feType.Int32:"int", feType.Int64:"bigint", feType.Float:"float", feType.Double:"double", feType.String:"string", feType.Date:"date", feType.Timestamp:"timestamp"}

table_schema = [
    ("engine_no", "int"),
    ("time_in_cycles", "int"),
    ("operational_setting_1", "double"),
    ("operational_setting_2", "double"),
    ("operational_setting_3", "double"),
    ("sensor_measurement_1", "double"),
    ("sensor_measurement_2", "double"),
    ("sensor_measurement_3", "double"),
    ("sensor_measurement_4", "double"),
    ("sensor_measurement_5", "double"),
    ("sensor_measurement_6", "double"),
    ("sensor_measurement_7", "double"),
    ("sensor_measurement_8", "double"),
    ("sensor_measurement_9", "double"),
    ("sensor_measurement_10", "double"),
    ("sensor_measurement_11", "double"),
    ("sensor_measurement_12", "double"),
    ("sensor_measurement_13", "double"),
    ("sensor_measurement_14", "double"),
    ("sensor_measurement_15", "double"),
    ("sensor_measurement_16", "double"),
    ("sensor_measurement_17", "double"),
    ("sensor_measurement_18", "double"),
    ("sensor_measurement_19", "double"),
    ("sensor_measurement_20", "double"),
    ("sensor_measurement_21", "double"),
    ("record_index", "int"),
    ("record_time", "timestamp"),
]

def get_schema():
    dict_schema = {}
    for i in table_schema:
        dict_schema[i[0]] = i[1]
    return dict_schema

dict_schema = get_schema()
json_schema = json.dumps(dict_schema)

class SchemaHandler(tornado.web.RequestHandler):
    def get(self):
        self.write(json_schema)

class PredictHandler(tornado.web.RequestHandler):
    def post(self):
        row = json.loads(self.request.body)
        data = {}
        for i in table_schema:
            if i[1] == "string":
                data[i[0]] = row.get(i[0], "")
            elif i[1] == "int" or i[1] == "double" or i[1] == "timestamp" or i[1] == "bigint":
                data[i[0]] = row.get(i[0], 0)
            else:
                data[i[0]] = None

        rs = connection.execute(sql, data)
        ins = pd.DataFrame()
        for r in rs:
            ins = ins.append([np.array(r).tolist()])
        
        # X['0'] is engine_no
        ins = ins.drop(0, axis=1)
        print(ins)
        self.write("----------------ins---------------\n")
        self.write(ins.to_string() + "\n")
        rul = reg.predict(ins)
        self.write("---------------predict rul -------------\n")
        print(rul)
        self.write(str(rul))

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("real time execute sparksql demo")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/schema", SchemaHandler),
        (r"/predict", PredictHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(9887)
    print("predict server started on port 9887")
    tornado.ioloop.IOLoop.current().start()
