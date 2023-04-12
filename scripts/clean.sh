#!/bin/bash

rm -rf data/unannotated/*
rm data/annotated/*
rm data/annotated/archive/*
rm data/annotated/new/*
rm data/metrics.json
rm data/run.log
rm data/progress.log

mkdir -p data/annotated/archive

