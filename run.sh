#!/bin/bash
python og_marl/tf2_systems/offline/maicq.py -m seed=1,2 task.source=alberdice task.env=rware task.scenario=tiny-2ag task.dataset=Expert
