#!/usr/bin/env bash

./build.sh

docker save hierarchialdet | gzip -c > HierarchialDet.tar.gz
