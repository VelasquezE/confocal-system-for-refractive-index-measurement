import os
from focus.analysis import find_focused_frames
from focus.criteria import compute_tenengrand, compute_laplacian, compute_sobel_variance

video_url = input("Enter video url: ")

folder_name = "LS4c_test"
os.makedirs(folder_name, exist_ok = "True")

find_focused_frames(video_url, compute_tenengrand, folder_name, 4)