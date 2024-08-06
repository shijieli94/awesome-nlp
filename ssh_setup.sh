#!/bin/bash

port=6001
node="lightning1"
test=false

service="ihpc.uts.edu.au"

while [[ $# -gt 0 ]]; do
  case $1 in
  --port)
    port="$2"
    shift 2
    ;;
  --node)
    node="$2"
    shift 2
    ;;
  --test)
    test=true
    shift
    ;;
  *)
    echo "Unknown argument $1"
    exit 1
    ;;
  esac
done

echo "port=$port node=$node"
if $test; then
  ssh -p $port shijli@localhost
else
  ssh -L $port:$node.$service:22 shijli@access.$service
fi
