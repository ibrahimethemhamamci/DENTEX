#!/usr/bin/env bash

./build.sh

docker save dentexchallenge | gzip -c > DentexChallenge.tar.gz
