import os
import numpy as np
from focus.analysis import find_focused_frames
from focus.criteria import compute_tenengrand, compute_laplacian, compute_sobel_variance

video_url = input("Enter video url: ")
basename = os.path.basename(video_url)
video_name = os.path.splitext(basename)[0]

folder_name = f"results/{video_name}"

os.makedirs(folder_name, exist_ok = "True")

text_name = f"results/summary_{video_name}.txt"
with open(text_name, "w", encoding = "utf-8") as f:
    f.write(f"Summary {video_name} \n")
    f.write("---------------------------------- \n")

methods = np.array([compute_tenengrand, compute_laplacian, compute_sobel_variance])

for criteria in methods:
    find_focused_frames(video_url, criteria, folder_name, text_name)