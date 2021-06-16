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

export JAVA_HOME=/home/jovyan/work/jdk1.8.0_141
export SPARK_HOME=/home/jovyan/work/spark-3.0.0-bin-sparkfe/
export PATH=$JAVA_HOME/bin:$PATH
export PATH=$SPARK_HOME/bin:$PATH
python3 train_sql.py
