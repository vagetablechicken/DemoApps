#! /bin/sh
#
# start_demo.sh

docker run -e GRANT_SUDO=yes --user root  -v `pwd`:/home/jovyan/work/rul -h=`hostname` --network=host -t 4pdosc/fedb_notebook:0.4.0 /bin/bash -c "cd /home/jovyan/work/rul && ls -la  && whoami && pip install -r demo/requirements.txt && jupyter notebook --allow-root"

# TODO(hw): add developer's guide. Normal user just use the image, no mount