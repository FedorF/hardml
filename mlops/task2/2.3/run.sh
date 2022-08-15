#!/bin/sh
docker run -d -p 8100:8000 \
-e REPLICA_NAME=web_app_replica1 \
-e REPLICA_PORT=8100 \
--rm \
task2.2

docker run -d -p 8200:8000 \
-e REPLICA_NAME=web_app_replica2 \
-e REPLICA_PORT=8200 \
--rm \
task2.2

docker run -d -p 8300:8000 \
-e REPLICA_NAME=web_app_replica3 \
-e REPLICA_PORT=8300 \
--rm \
task2.2
