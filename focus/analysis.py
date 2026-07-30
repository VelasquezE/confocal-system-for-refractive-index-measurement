import numpy as np
import cv2
from scipy.signal import find_peaks, peak_prominences
from . import utils

from collections.abc import Callable

def measure_focus(video, method: Callable):
    """
    Computes the focus score for each frame of the video.
    Parameters:
        video (cv2.VideoCapture).
        method (Callable): Function that'll be used to
        calculate the focus score.
    Returns:
        focus_scores (np.ndarray): Array of floats.
        frames (np.ndarray): Array of frames. 
    """
    video_info = utils.get_video_info(video)

    focus_scores = np.zeros(video_info["total_frames"])
    frames = np.zeros((video_info["total_frames"], video_info["height"], 
                       video_info["width"]), dtype = np.float64)

    counter = 0
    
    while True:
        ret, frame = video.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            focus_scores[counter] = method(gray_frame)
            frames[counter] = gray_frame
            counter += 1
        else:
            if counter < (video_info["total_frames"] - 1):
                raise VideoReadError("Error occurred.")
            else:
                break 
            
    video.release()
    
    return focus_scores, frames



def identify_peaks(focus_scores: np.ndarray) -> np.ndarray:
    """
    Uses the function find_peaks to find the values corresponding to the peaks
    of the graph. Then, calculates the prominence of each peak. Finally, sorts
    the prominence array and select the two peaks with higher prominences. 
    Parameters:
        focus_scores (np.ndarray): Array of floats.
    Returns:
        peaks (np.ndarray): Frame index of the focus scores peaks.
    """
    all_peaks, properties = find_peaks(focus_scores)
    prominences = peak_prominences(focus_scores, all_peaks)[0]
    peaks_with_prominence = dict(zip(all_peaks, prominences))
    sorted_prominences = np.sort(prominences)
    peaks = [key for key, value in peaks_with_prominence.items() if value in sorted_prominences[-2:]]

    return peaks

def find_focused_frames(name, method, folder_name, text_name):
    """
    Gets the focus scores of the video, prints the peaks value, plots the individual frames, 
    plots a summary graph, plots the scores.
    """
    try:
        video = utils.open_video(name)
        print("Video file opened successfully!")
    except FileNotFoundError as e:
        print("Error:", e)
    
    focus_scores, frames = measure_focus(video, method)
    index_focused_frames = identify_peaks(focus_scores)

    print(f"Los máximos están en los frames: {index_focused_frames}")

    methods = {"compute_tenengrand": "Tenengrand", "compute_sobel_variance": "SobelVariance",
              "compute_laplacian": "Laplacian"}
    function_name = method.__name__
    method_name = methods[function_name]

    with open(text_name, "a", encoding = "utf-8") as file:
        file.write(f"{method_name} seleccionó los frames {index_focused_frames} \n")
    
    utils.plot_focused_frames(frames, index_focused_frames, method_name, folder_name)
    utils.plot_scores(focus_scores, method_name, folder_name)