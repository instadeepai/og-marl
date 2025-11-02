DATASET_URLS = {
    "smac_v1": {
        "3m": "https://tinyurl.com/3m-dataset",
        "8m": "https://tinyurl.com/8m-dataset",
        "5m_vs_6m": "https://tinyurl.com/5m-vs-6m-dataset",
        "2s3z": "https://tinyurl.com/2s3z-dataset",
        "3s5z_vs_3s6z": "https://tinyurl.com/3s5z-vs-3s6z-dataset3",
        "2c_vs_64zg": "https://tinyurl.com/2c-vs-64zg-dataset",
        "27m_vs_30m": "https://tinyurl.com/27m-vs-30m-dataset",
    },
    "smac_v2": {
        "terran_5_vs_5": "https://tinyurl.com/terran-5-vs-5-dataset",
        "zerg_5_vs_5": "https://tinyurl.com/zerg-5-vs-5-dataset",
        "terran_10_vs_10": "https://tinyurl.com/terran-10-vs-10-dataset",
    },
    "flatland": {
        "3_trains": "https://tinyurl.com/3trains-dataset",
        "5_trains": "https://tinyurl.com/5trains-dataset",
    },
    "pettingzoo": {
        "pursuit": "https://tinyurl.com/pursuit-dataset",
        "pistonball": "https://tinyurl.com/pistonball-dataset",
        "coop_pong": "https://tinyurl.com/coop-pong-dataset",
        "kaz": "https://tinyurl.com/kaz-dataset",
    },
    "mamujoco": {
        "2_halfcheetah": "https://tinyurl.com/2halfcheetah-dataset",
        "2_ant": "https://tinyurl.com/2ant-dataset",
        "4_ant": "https://tinyurl.com/4ant-dataset",
    },
    "voltage_control": {
        "case33_3min_final": "https://tinyurl.com/case33-3min-final-dataset",
    },
    "city_learn": {"2022_all_phases": "https://tinyurl.com/2022-all-phases-dataset"},
    "mpe": {"simple_adversary": "https://tinyurl.com/simple-adversary-dataset"},
}

env_names = [
    "SMAC V1",
    "SMAC V2",
    "Flatland",
    "PettingZoo",
    "MaMuJoCo",
    "Voltage Control",
    "City Learn",
    "MPE",
]

lines = [
    "<!DOCTYPE html>\n",
    '<html lang="en">\n',
    '<meta charset="UTF-8">\n',
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n',
    "<title>Website Page</title>\n",
    "</head>\n",
    "<body>\n",
    "<h1>Datasets</h1>\n",
]
for i, env in enumerate(list(DATASET_URLS.keys())):
    lines.extend(["<h2>" + env_names[i] + "</h2>\n", "<ul>\n"])
    for mapping in DATASET_URLS[env].keys():
        lines.append("  <li><a href=" + DATASET_URLS[env][mapping] + ">" + mapping + "</a></li>\n")
    lines.extend(["</ul>\n", "\n"])

lines.extend(["</body>\n", "</html>\n"])

f = open("test.txt", "w")
f.writelines(lines)
f.close()
