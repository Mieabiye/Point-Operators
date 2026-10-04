import cv2
import numpy as np

def cv2_imshow(image, window_name="Image"):
    """Display an image using a normal OpenCV window (local equivalent of Colab's cv2_imshow)."""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#   IMAGE PATH 
IMAGE_PATH = r"C:\Users\HP\Documents\Unmanned Devices and Processes\Point Operators\flower.jpg"

def _load_image(image_path):
    """Loads an image using cv2.imread."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    return image

def _clip_image(image):
    """Clips values to [0, 255] and converts back to uint8."""
    return np.clip(image, 0, 255).astype(np.uint8)


# 1. CONTRAST

def contrast(image_path, alpha=1.5, beta=0.0, save_path=None, show=True):
    """
    Adjusts contrast of a color image across all three RGB (BGR) channels.
    Formula: new_pixel = alpha * (pixel - 128) + 128 + beta
    """
    image = _load_image(image_path).astype(np.float32)
    result = alpha * (image - 128.0) + 128.0 + beta
    result = _clip_image(result)

    if save_path:
        cv2.imwrite(save_path, result)
    if show:
        cv2_imshow(result, "Contrast")

    return result


# 2. BRIGHTNESS

def brightness(image_path, beta=50, save_path=None, show=True):
    """
    Adjusts brightness of a color image across all three channels.
    Formula: new_pixel = pixel + beta
    """
    image = _load_image(image_path).astype(np.float32)
    result = image + beta
    result = _clip_image(result)

    if save_path:
        cv2.imwrite(save_path, result)
    if show:
        cv2_imshow(result, "Brightness")

    return result


# 3. INVERSION

def inversion(image_path, save_path=None, show=True):
    """
    Inverts colors of a color image (photographic negative).
    Formula: new_pixel = 255 - pixel
    """
    image = _load_image(image_path).astype(np.float32)
    result = 255.0 - image
    result = _clip_image(result)

    if save_path:
        cv2.imwrite(save_path, result)
    if show:
        cv2_imshow(result, "Inversion")

    return result


# 4. BLENDING

def blending(image_path1, image_path2, alpha=0.5, save_path=None, show=True):
    """
    Blends two color images together using a weighted sum.
    Formula: new_pixel = alpha * img1 + (1 - alpha) * img2
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in the range [0, 1]")

    img1 = _load_image(image_path1).astype(np.float32)
    img2 = _load_image(image_path2).astype(np.float32)

    # Match the second image to the first image's size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)

    result = alpha * img1 + (1.0 - alpha) * img2
    result = _clip_image(result)

    if save_path:
        cv2.imwrite(save_path, result)
    if show:
        cv2_imshow(result, "Blending")

    return result


#   EXAMPLES

if __name__ == "__main__":

    print("Running contrast...")
    contrast(IMAGE_PATH, alpha=1.8, beta=0.0,
             save_path=r"C:\Users\HP\Documents\Unmanned Devices and Processes\Point Operators\flower_contrast.jpg")

    print("Running brightness...")
    brightness(IMAGE_PATH, beta=60,
               save_path=r"C:\Users\HP\Documents\Unmanned Devices and Processes\Point Operators\flower_bright.jpg")

    print("Running inversion...")
    inversion(IMAGE_PATH,
              save_path=r"C:\Users\HP\Documents\Unmanned Devices and Processes\Point Operators\flower_inverted.jpg")

    # Blending needs a second image — change or comment out this block.
    SECOND_IMAGE = r"C:\Users\HP\Documents\Unmanned Devices and Processes\Point Operators\flower2.jpg"
    try:
        print("Running blending...")
        blending(IMAGE_PATH, SECOND_IMAGE, alpha=0.5,
                 save_path=r"C:\Users\HP\Documents\Unmanned Devices and Processes\Point Operators\flower_blended.jpg")
    except FileNotFoundError as e:
        print(f"Skipping blending: {e}")

    print("Done! Check your Point Operators folder for the output files.")