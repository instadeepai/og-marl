#!/bin/bash
python og_marl/tf2_systems/online/ic51.py seed=4 task.scenario=5m_vs_6m 
python og_marl/tf2_systems/online/iqrdqn.py seed=4 task.scenario=5m_vs_6m
python og_marl/tf2_systems/online/iql.py seed=4 task.scenario=5m_vs_6m