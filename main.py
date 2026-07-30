import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from focus.utils import open_video, get_video_info
from focus.analysis import find_focused_frames
from focus.criteria import compute_tenengrand, compute_laplacian, compute_sobel_variance
from calculate.refraction_index import calculate_apparent_depth, get_refraction_index

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

custom_rc = {
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.edgecolor": "black",
    "grid.color":"#bbbbbb",
    "grid.linestyle": "dashed",
    "axes.grid": True,
    "axes.spines.top": True,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True
}

video_url = input("Enter video url: ")

try:
    video = open_video(video_url)
    print("Video file opened successfully!")
except FileNotFoundError as e:
        print("Error:", e)

video_info = get_video_info(video)
basename = os.path.basename(video_url)
video_name = os.path.splitext(basename)[0]

folder_name = f"results/{video_name}"

os.makedirs(folder_name, exist_ok = "True")

text_name = f"{folder_name}/summary_{video_name}.txt"
with open(text_name, "w", encoding = "utf-8") as f:
    f.write(f"Summary {video_name} \n")
    f.write("---------------------------------- \n")

methods = np.array([compute_tenengrand, compute_laplacian, compute_sobel_variance])

#car_velocity = 0.2 # mm/s
#t_lens = 4.24 # mm
t_lens = float(input("Ingrese el espesor del objeto óptico: "))
car_velocity = float(input("Ingrese la velocidad del carro: "))

for criteria in methods:
    index_focused_frames = find_focused_frames(video_url, criteria, folder_name, text_name)
    apparent_depth = calculate_apparent_depth(index_focused_frames, video_info["fps"], car_velocity)
    refraction_index = get_refraction_index(t_lens, apparent_depth)
    with open(text_name, "a", encoding = "utf-8") as file:
        file.write(f"y calculó n = {refraction_index:0.4f}.\n")
