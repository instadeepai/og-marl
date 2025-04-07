#!/bin/bash
python og_marl/tf2_systems/offline/maicq.py -m seed=9,10 task.source=alberdice task.env=rware task.scenario=small-2ag task.dataset=Expert
