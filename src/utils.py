"""
工具模块 — 点云生成 + 类别定义
"""
import numpy as np
from pathlib import Path
from typing import Optional


def normalize_points(points):
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points


def sample_points(points, num_points):
    N = points.shape[0]
    if N >= num_points:
        indices = np.random.choice(N, num_points, replace=False)
    else:
        indices = np.random.choice(N, num_points, replace=True)
    return points[indices]


SHAPES = ["sphere", "cube", "cylinder", "cone", "chair", "table", "airplane"]


def generate_shape(shape, num_points=1024):
    if shape == "sphere":
        phi = np.random.uniform(0, np.pi, num_points)
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points = np.stack([x, y, z], axis=1)
    elif shape == "cube":
        points = np.random.uniform(-1, 1, (num_points, 3))
    elif shape == "cylinder":
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        r = np.ones(num_points) * 0.8
        z = np.random.uniform(-1, 1, num_points)
        points = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    elif shape == "cone":
        z = np.random.uniform(0, 1.5, num_points)
        r = 1.0 - z / 1.5
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        points = np.stack([r * np.cos(theta), r * np.sin(theta), z - 0.75], axis=1)
    elif shape == "chair":
        seat = np.random.uniform(-0.5, 0.5, (num_points // 4, 3))
        seat[:, 2] = 0.5
        back = np.random.uniform(-0.5, 0.5, (num_points // 4, 3))
        back[:, 1] = 0.5
        back[:, 2] = np.random.uniform(0.5, 1.5, num_points // 4)
        leg1 = np.random.uniform(-0.05, 0.05, (num_points // 8, 3))
        leg1[:, 0] -= 0.45; leg1[:, 1] -= 0.45; leg1[:, 2] = np.random.uniform(0, 0.5, num_points // 8)
        leg2 = np.random.uniform(-0.05, 0.05, (num_points // 8, 3))
        leg2[:, 0] += 0.45; leg2[:, 1] -= 0.45; leg2[:, 2] = np.random.uniform(0, 0.5, num_points // 8)
        leg3 = np.random.uniform(-0.05, 0.05, (num_points // 8, 3))
        leg3[:, 0] -= 0.45; leg3[:, 1] += 0.45; leg3[:, 2] = np.random.uniform(0, 0.5, num_points // 8)
        leg4 = np.random.uniform(-0.05, 0.05, (num_points // 8, 3))
        leg4[:, 0] += 0.45; leg4[:, 1] += 0.45; leg4[:, 2] = np.random.uniform(0, 0.5, num_points // 8)
        points = np.vstack([seat, back, leg1, leg2, leg3, leg4])
        points = sample_points(points, num_points)
    elif shape == "table":
        top = np.random.uniform(-0.6, 0.6, (num_points // 3, 3))
        top[:, 2] = 0.7
        leg1 = np.random.uniform(-0.05, 0.05, (num_points // 6, 3))
        leg1[:, 0] -= 0.5; leg1[:, 1] -= 0.5; leg1[:, 2] = np.random.uniform(0, 0.7, num_points // 6)
        leg2 = np.random.uniform(-0.05, 0.05, (num_points // 6, 3))
        leg2[:, 0] += 0.5; leg2[:, 1] -= 0.5; leg2[:, 2] = np.random.uniform(0, 0.7, num_points // 6)
        leg3 = np.random.uniform(-0.05, 0.05, (num_points // 6, 3))
        leg3[:, 0] -= 0.5; leg3[:, 1] += 0.5; leg3[:, 2] = np.random.uniform(0, 0.7, num_points // 6)
        leg4 = np.random.uniform(-0.05, 0.05, (num_points // 6, 3))
        leg4[:, 0] += 0.5; leg4[:, 1] += 0.5; leg4[:, 2] = np.random.uniform(0, 0.7, num_points // 6)
        points = np.vstack([top, leg1, leg2, leg3, leg4])
        points = sample_points(points, num_points)
    elif shape == "airplane":
        fuselage = np.random.uniform(-0.1, 0.1, (num_points // 2, 3))
        fuselage[:, 0] = np.random.uniform(-1.0, 1.0, num_points // 2)
        wing = np.random.uniform(-0.1, 0.1, (num_points // 4, 3))
        wing[:, 1] = np.random.uniform(-1.0, 1.0, num_points // 4)
        wing[:, 0] = np.random.uniform(-0.2, 0.2, num_points // 4)
        tail = np.random.uniform(-0.1, 0.1, (num_points // 4, 3))
        tail[:, 0] = np.random.uniform(0.8, 1.0, num_points // 4)
        tail[:, 2] = np.random.uniform(0, 0.4, num_points // 4)
        points = np.vstack([fuselage, wing, tail])
        points = sample_points(points, num_points)
    else:
        points = np.random.uniform(-1, 1, (num_points, 3))

    return normalize_points(points).astype(np.float32)


MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]
