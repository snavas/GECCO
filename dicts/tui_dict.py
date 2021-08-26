import numpy as np

tui_dict = {
    7: {
        "color": (0,0,0),
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8'),
        "inside": 1.0
    }, # eraser
    1: {
        "color": (3, 200, 3),
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8'),
        "inside": 1.0
    }, # black pen
    2: {
        "color": (3, 3, 200),
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8'),
        "inside": 1.0
    },
    3: {
        "color": (3, 200, 200),
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8'),
        "inside": 1.0
    },
    4: {
        "color": (200, 3, 3),
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8'),
        "inside": 1.0
    },
    5: {
        "color": (255, 255, 255),
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8'),
        "inside": 1.0
    },
    6: {
        "color": (200, 3, 200),
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8'),
        "inside": 1.0
    },
    8: {
        "thickness": 2,
        "edges": np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [0, 0]]], dtype='uint8'),
        "inside": 1.0
    }
}