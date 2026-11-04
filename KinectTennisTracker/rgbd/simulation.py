from array import array
from dataclasses import dataclass
from random import Random

from .frames import Frame, Vec3

FPS = 30
WIDTH, HEIGHT = 192, 144
FOCAL = 106
SCENARIOS = {"single": 90, "crossing": 120, "ballistic": 30}


@dataclass(frozen=True)
class Truth:
    xyz: Vec3
    velocity: Vec3


def ground_truth(scenario: str, index: int) -> dict[int, Truth]:
    t = index / FPS
    if scenario == "ballistic":
        return {1: Truth((-.45 + .94 * t, -.53 + 4.9 * t - 4.905 * t**2, 2.2),
                         (.94, 4.9 - 9.81 * t, 0.0))}
    objects = {1: Truth((-.66 + .34 * t, .14, 1.9 + .06 * t), (.34, 0.0, .06))}
    if scenario == "crossing":
        objects.update({
            2: Truth((.68 - .34 * t, -.19, 2.27 - .06 * t), (-.34, 0.0, -.06)),
            3: Truth((.10, .42 - .04 * t, 2.44 - .10 * t), (0.0, -.04, -.10)),
        })
    return objects


def simulate(scenario: str, seed: int = 7):
    rng = Random(seed)
    pixels = WIDTH * HEIGHT
    background = bytes((37, 37, 37, 0)) * pixels
    for index in range(SCENARIOS[scenario]):
        color = bytearray(background)
        xyz = array("f", [0.0]) * (pixels * 3)
        depth = array("H", [0]) * pixels
        truth = ground_truth(scenario, index)
        for object_id, state in sorted(truth.items(), key=lambda item: -item[1].xyz[2]):
            if scenario == "crossing" and object_id == 1 and 42 <= index <= 48:
                continue
            x, y, z = state.xyz
            u = WIDTH / 2 + FOCAL * x / z + rng.gauss(0, .45)
            v = HEIGHT / 2 - FOCAL * y / z + rng.gauss(0, .45)
            radius = FOCAL * .115 / z
            z_noisy = z + rng.gauss(0, .009)
            no_depth = scenario == "crossing" and object_id == 3 and 63 <= index <= 67
            for py in range(max(0, int(v - radius)), min(HEIGHT, int(v + radius) + 1)):
                for px in range(max(0, int(u - radius)), min(WIDTH, int(u + radius) + 1)):
                    if (px - u)**2 + (py - v)**2 > radius**2:
                        continue
                    i = py * WIDTH + px
                    color[4 * i:4 * i + 4] = bytes((20, 110, 242, 0))
                    if no_depth or rng.random() < .08:
                        depth[i] = 0
                        xyz[3 * i] = xyz[3 * i + 1] = xyz[3 * i + 2] = 0
                        continue
                    depth[i] = round(z_noisy * 1000)
                    xyz[3 * i] = (px - WIDTH / 2) * z_noisy / FOCAL
                    xyz[3 * i + 1] = (HEIGHT / 2 - py) * z_noisy / FOCAL
                    xyz[3 * i + 2] = z_noisy
        stamp = round(index * 1000 / FPS)
        yield Frame(WIDTH, HEIGHT, WIDTH, HEIGHT, stamp, stamp, bytes(color),
                    memoryview(depth), memoryview(xyz)), truth
