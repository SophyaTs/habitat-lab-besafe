from habitat.core.utils import Singleton

from habitat.core.simulator import VisualObservation

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import attr

import base64

import json

import numpy as np

import cv2

import os


def display_grayscale(image):
    img_bgr = np.repeat(image, 3, 2)
    cv2.imshow("Depth Sensor", img_bgr)
    return cv2.waitKey(0)


def display_rgb(image):
    img_bgr = image[..., ::-1]
    cv2.imshow("RGB", img_bgr)
    return cv2.waitKey(0)


DISPLAY_FINAL_IMAGE = False
DISPLAY_DEPTH = False


@attr.s(auto_attribs=True, slots=True)
class ObservationsSingleton(metaclass=Singleton):
    buffers_to_get: List[str] = attr.ib(init=False, factory=list)
    buffers: Dict[str, Union[VisualObservation, bool, List[float]]] = attr.ib(
        init=False, factory=dict
    )

    def set_buffers(self, buffers: List[str]):
        # TODO implement this
        self.buffers_to_get = buffers

    def parse_buffers(self, obj):
        self.buffers = {}
        for key, value in obj.items():
            # print(f"Got buffer {key}")

            if key.startswith("_"):
                # not an image
                if key in ["_location", "_rotation"]:
                    self.buffers[key] = [float(i) for i in value.split(" ")]
                elif key == "_previous_step_reset":
                    self.buffers[key] = value == "True"
                elif key == "_previous_step_reset_reason":
                    self.buffers[key] = value
                else:
                    print(f"Unknown Observation {key} = {value}")
                    exit()
                continue

            image = base64.b64decode(value)

            # if key == "FinalImage":
            #    f = open(f"{key}.jpg", "wb")
            # else:
            #    f = open(f"{key}.png", "wb")
            # f.write(image)
            # f.close()

            image = np.asarray(bytearray(image), dtype="uint8")

            # use imdecode function
            if "Depth" in key:
                image = cv2.imdecode(image, cv2.IMREAD_ANYDEPTH)

                # print(f"depth has min={np.min(image)} and max={np.max(image)}")

                if np.min(image) < 0 or np.max(image) > 2**16 - 1:
                    print("Depth image out of range")
                    exit()

                image = image / (2**16 - 1)

                # Fit habitat's data interface
                image = np.expand_dims(image, axis=2)

                # image is saved as float64, but the habitat_baselines evaluator crashes, have to convert
                image = np.float32(image)

                # print(f"Depth has type: {type(image)} and {image.dtype}")
            else:
                image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # correctly layout data
            self.buffers[key] = image

            if DISPLAY_FINAL_IMAGE and key == "FinalImage":
                display_rgb(image)
            elif DISPLAY_DEPTH and "Depth" in key:
                display_rgb(image)

        # print(f"Got observations: {', '.join([k for k in obj.keys()])}")
        if (
            self.buffers["_previous_step_reset"]
            and self.buffers["_previous_step_reset_reason"] != "Habitat Reset"
        ):
            print(
                f"Previous action resulted in a reset because of: {self.buffers['_previous_step_reset_reason']}"
            )

    def __getattr__(self, name):
        return self.buffers[name]

    def __getitem__(self, name):
        return self.buffers[name]


Observations: ObservationsSingleton = ObservationsSingleton()
