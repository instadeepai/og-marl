import numpy as np
from dataclasses import dataclass

DIRECTION_UP = 0
DIRECTION_DOWN = 1
DIRECTION_LEFT = 2
DIRECTION_RIGHT = 3


@dataclass
class NearInformation:
    x: int
    y: int
    is_agent: bool
    agent_direction: int
    is_shelf: bool
    is_requested_shelf: bool


@dataclass
class Observation:
    x: int
    y: int
    is_carrying: bool
    direction: int
    is_path_location: bool
    near_info: list


class ObservationParser:

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @staticmethod
    def parse(obs):
        parsed_obs = Observation(
            x=obs[0], 
            y=obs[1], 
            is_carrying=obs[2] == 1.0, 
            direction=int(np.argmax(obs[3:7])),
            is_path_location=obs[7] == 1.0, 
            near_info=ObservationParser.parse_near_info(obs)
        )
        return parsed_obs

    @staticmethod
    def parse_near_info(obs):
        agent_x = obs[0]
        agent_y = obs[1]

        near_info = []
        infos = list(ObservationParser.chunks(obs[8:], 7))
        
        for i, info in enumerate(infos):
            row = i // 3
            col = i % 3
            near_info.append(NearInformation(
                x=agent_x - 1 + row,
                y=agent_y - 1 + col,
                is_agent=info[0] == 1.0,
                agent_direction=int(np.argmax(info[1:5])),
                is_shelf=info[5] == 1.0, 
                is_requested_shelf=info[6] == 1.0
            ))

        return near_info
