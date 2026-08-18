#!/bin/sh
set -e

xvfb-run -a python /app/scripts/run_ea.py "$@"