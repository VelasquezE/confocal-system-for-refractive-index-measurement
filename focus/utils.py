import cv2
import matplotlib.pyplot as plt

def open_video(video_url: str):
    """
    Uploads the video given its url. 
    Parameters:
        video_url (str): path of the video.
    Returns:
        video (cv2.VideoCapture object).
    """
    video = cv2.VideoCapture(video_url)
    
    if not video.isOpened():
        raise FileNotFoundError("No se pudo abrir el video")
        
    return video

def get_video_info(video)-> dict:
    """
    Creates a dictionary with the basic information
    of the video.
    Parameters:
        video (cv2.VideoCapture object).
    Returns:
        video_info (dict): Dictionary with fps, total frames, 
        widht, and height of the video.
    """
    video_info = {}
    video_info["fps"] = video.get(cv2.CAP_PROP_FPS)
    video_info["total_frames"] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_info["width"] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_info["height"] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return video_info

def plot_focused_frames(frames, focused_index, method_name, folder_name): 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(frames[focused_index[0]], cmap = "gray")
    ax2.imshow(frames[focused_index[1]], cmap = "gray")
    ax1.set_title(f"Frame {focused_index[0]}")
    ax2.set_title(f"Frame {focused_index[1]}")
    ax1.axis('off')
    ax2.axis('off')
    plt.suptitle(method_name)
    plt.tight_layout()
    fig.savefig(f"{folder_name}/{method_name}_resumen.png", dpi = 300, bbox_inches = "tight")

def plot_individual_frames(frame, frame_index, method_name, folder_name):
    fig, ax = plt.subplots(1, 1)
    
    ax.imshow(frame, cmap = "gray")
    ax.axis("off")
    fig.savefig(f"{folder_name}/{method_name}_{frame_index}.png", dpi = 300, bbox_inches = "tight")


def plot_scores(scores, method_name, folder_name):
    focus_dictionary = {ii: float(score) for ii, score in enumerate(scores)}

    fig, ax = plt.subplots(1,1)
    ax.plot(focus_dictionary.keys(), focus_dictionary.values())
    ax.set_title(f"{method_name}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Puntaje de enfoque")
    fig.savefig(f"{folder_name}/{method_name}_puntaje.png", dpi = 300, bbox_inches = "tight")