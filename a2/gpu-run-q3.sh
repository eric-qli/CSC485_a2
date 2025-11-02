#!/usr/bin/env bash

srun -p csc485 --gres gpu -c 2 python3.12 q3.py
