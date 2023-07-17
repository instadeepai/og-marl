"""Score normilisation"""

def normalise_score(score, env, task):
    env = env.lower()
    task = task.lower()
    min_score = (
        RANDOM_SCORES[env][task]
        if type(RANDOM_SCORES[env]) is dict
        else RANDOM_SCORES[env]
    )
    max_score = (
        EXPERT_SCORES[env][task]
        if type(EXPERT_SCORES[env]) is dict
        else EXPERT_SCORES[env]
    )
    return (score - min_score) / (max_score - min_score)

RANDOM_SCORES = {
    "smac": {
        "3m": 1.47,
        "8m": 1.83,
        "2s3z": 3.95,
        "5m_vs_6m": 1.41,
        "2c_vs_64zg": 9.76,
        "3s5z_vs_3s6z": 3.9,
        "27m_vs_30m": 1.94,
    },
    "mamujoco": {
        "2halfcheetah": -280.18,
        "2ant": -325.6,
        "4ant": -325.6,
    },
    "pettingzoo": {
        "pursuit": -46.25,
        "coop_pong": -2.88,
        "pistonball": 3.82,
    },
    "flatland": {"5trains": -31.95, "3trains": -32.07},
}

MEAN_SCORES = {
    "smac": {
        "3m": {"poor": 4.85, "medium": 10.05, "good": 16.02},
        "8m": {"poor": 5.27, "medium": 10.28, "good": 16.27},
        "2s3z": {"poor": 6.84, "medium": 12.84, "good": 18.21},
        "5m_vs_6m": {"poor": 7.65, "medium": 12.85, "good": 16.58},
        "2c_vs_64zg": {"poor": 9.88, "medium": 13.1, "good": 17.97},
        "3s5z_vs_3s6z": {"poor": 5.71, "medium": 11.05, "good": 16.99},
        "27m_vs_30m": {"poor": 5.68, "medium": 10.55, "good": 15.99},
    },
    "mamujoco": {
        "2halfcheetah": {"poor": 400.45, "medium": 1485, "good": 6924.11},
        "2ant": {"poor": 437.67, "medium": 1099.98, "good": 2621.5},
        "4ant": {"poor": 542.73, "medium": 1546.05, "good": 2769.29},
    },
    "pettingzoo": {
        "pursuit": {"poor": -27.35, "medium": 22.66, "good": 79.45},
        "coop_pong": {"poor": 14.36, "medium": 35.63, "good": 65.13},
        "pistonball": {"poor": 12.04, "medium": 34.14, "good": 84.61},
    },
    "flatland": {
        "5trains": {"poor": -25.53, "medium": -16.34, "good": -5.94},
        "3trains": {"poor": -28.76, "medium": -16.14, "good": -5.16},
    },
}

EXPERT_SCORES = {
    "smac": 20.,
    "mamujoco": {
        "2halfcheetah": 9132.25,
        "2ant": 3362.44,
        "4ant": 3224.91,
    },
    "pettingzoo": {"pursuit": 124.22, "coop_pong": 100., "pistonball": 100.},
    "flatland": {
        "5trains": 0.,
        "3trains": 0.,
    },
}