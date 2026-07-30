import numpy as np

def calculate_apparent_depth(index_focused_frames: np.ndarray, video_velocity: float,
                             car_velocity: float) -> float:
    """
    Calculates the apparent depth of the surface. 
    """
    frames_difference = index_focused_frames[1] - index_focused_frames[0]
    factor = car_velocity / video_velocity

    return frames_difference * factor


def get_refraction_index(t_lens: float, apparent_depth: float) -> float:
    """
    Calculates the refraction index. 
    Parameters:
        t_lens (float): Surface thickness (mm).
        apparent_depth (float)
    """
    return t_lens/apparent_depth