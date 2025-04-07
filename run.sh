#!/bin/bash
python og_marl/tf2_systems/offline/maicq.py -m seed=3,4 task.source=alberdice task.env=rware task.scenario=small-2ag task.dataset=Expert
