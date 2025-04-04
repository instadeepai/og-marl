#!/bin/bash
python og_marl/tf2_systems/offline/maicq.py -m seed=9,10 task.source=alberdice task.env=rware task.scenario=tiny-6ag task.dataset=Expert
