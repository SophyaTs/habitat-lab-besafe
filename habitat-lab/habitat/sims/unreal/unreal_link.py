# TCP client implementation
import socket
import struct
import json
import base64
import copy

from habitat.sims.unreal.observations import Observations

from habitat.sims.unreal.actions import UnrealSimActions

from omegaconf import OmegaConf


class UnrealLink:
    def __init__(self, ip="127.0.0.1") -> None:
        self.ip = ip  # "100.75.90.104"  # tailscale home machine
        self.port = 8890
        self.packet_size = 4096

    def connect_server(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.ip, self.port))
        self.client.settimeout(10)

    def close_connection(self):
        self.client.close()

    async def __receive_packet(self):
        try:
            size_packet = self.client.recv(4)
            size = struct.unpack("<i", size_packet)[0]

            next_recv = min(self.packet_size, size)
            response = self.client.recv(next_recv)
            while len(response) < size:
                next_rcv = min(self.packet_size, size - len(response))
                response += self.client.recv(next_rcv)

            decoded = response.decode()
            # print(f"Received {len(response)} bytes")

            return decoded
        except socket.timeout:
            print("Timed out, trying to recover")
            # instead of failing, ask for new observations,
            # assuming the action was performed?
            return await self.__send_packet("capture")

    async def __send_packet(self, payload):
        self.client.send(payload.encode())

        # always await a response
        response = await self.__receive_packet()

        return response

    def __handle_observations(self, observation):
        # Testing if it's a json... yeah
        if observation[0] == "{":
            obj = json.loads(observation)
            Observations.parse_buffers(obj)
        else:
            print(observation)

    async def execute_action(self, action):
        # TODO error check? make new json field to detect errors or stop?
        action_name = UnrealSimActions.get_unreal_action(action)

        # print(f"Executing action {action_name}")

        observation = await self.__send_packet(f"action {action_name}")

        self.__handle_observations(observation)

    async def capture_observation(self):
        # TODO error check? make new json field to detect errors or stop?
        observation = await self.__send_packet("capture")

        self.__handle_observations(observation)

    async def reset_environment(self):
        observation = await self.__send_packet(f"reset")

        self.__handle_observations(observation)

    async def submit_settings(self, config):
        settings = json.dumps(OmegaConf.to_container(config))
        result = await self.__send_packet(settings)

        if result == "OK":
            pass
        else:
            print(f"Unreal server didn't accept the settings! {result}")
            print(f"Sent the payload: {settings}")
            exit()

    async def begin_simulation(self, start_location, start_rotation):
        # TODO error check? make new json field to detect errors or stop?
        print(f"Beginning the simulation")

        #TODO change rotation
        # rotation = convert_to_unreal_rotation(start_rotation)
        result = await self.__send_packet(
            f"begin_sim {' '.join(map(str, start_location))} {' '.join(map(str, start_rotation))}"
        )

        # print(
        #    f"changing episode to {' '.join(map(str, start_location))} {' '.join(map(str, start_rotation))}"
        # )

        if result == "OK":
            pass
        else:
            print(
                f"Unreal server didn't accept the start location/rotations! {result}"
            )
            exit()

    async def query_geodesic_distance(self, point_a, point_b):
        # TODO error check? make new json field to detect errors or stop?

        payload = f"geodesic_distance {' '.join(map(str, point_a))} {' '.join(map(str, point_b))}"
        queried_distance = await self.__send_packet(payload)
        # print(f"{payload=}")

        try:
            distance = float(queried_distance)
            if distance == -1.0:
                raise Exception("Invalid path!")
            return distance
        except Exception as e:
            print(f"Could not compute geodesic distance! {e}")

    async def query_closest_obstacle_distance(
        self, position, max_detection_radius
    ):
        # TODO error check? make new json field to detect errors or stop?
        payload = f"closest_obstacle_distance {' '.join(map(str, position))} {str(max_detection_radius)}"
        queried_distance = await self.__send_packet(payload)
        # print(f"{payload=}")

        try:
            distance = float(queried_distance)
            if distance == -1.0:
                raise Exception("Invalid query!")
            return distance
        except Exception as e:
            print(f"Could not compute distance to closest obstacle! {e}")
