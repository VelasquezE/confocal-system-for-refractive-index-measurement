import numpy as np
import cv2


def compute_tenengrand(image: np.ndarray) -> float:
    """
    Computes the gradient with the sobel operator in each direction 
    and returns the mean.
    Parameters:
        image (np.ndarray): frame that's going to be evaluated.        
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3)
    tenengrand = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    return np.mean(tenengrand)

def compute_sobel_variance(image):
    """
    Computes the gradient with the sobel operator in each direction 
    and the intensity variance.
    Parameters:
        image (np.ndarray): frame that's going to be evaluated.        
    """

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2) 
    variance = np.var(image)  
    
    return np.mean(sobel_magnitude) + variance

def compute_laplacian(image):
    """
    Uses the Laplacian operator and gets the variance of its
    response. 
    
    Parameters:
        image (np.ndarray): frame that's going to be evaluated.        
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)  
    return np.var(laplacian)
