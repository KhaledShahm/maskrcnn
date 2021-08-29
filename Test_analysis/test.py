import sys
import os
import numpy as np 
import cv2
import logging

from skimage import morphology
import yaml 
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torchvision.transforms import functional as F


using_logs = False


morphology_images_dir = os.path.join('images')
logs_dir = "logs"




def init_logs():
    """creates a new log file and sets the global variabel "using_log" to True
    """

    global using_logs
    logFileName = os.path.join(logs_dir, "log_" + str(len(os.listdir(logs_dir))) + ".log" )
    print("initializing logs at:", logFileName)
    logging.basicConfig(level=logging.DEBUG, filename= logFileName, filemode = "w", format = '%(levelname)-8s %(asctime)s %(message)s')
    using_logs = True




def log(message, logType = 'DEBUG', and_print = False, only_print = False, nothing = False, urgent = False):
    """logs a message

    Args:
        message (str): the message to be logged
        logType (str, optional): can be 'info', 'warn' or 'debug'. Determines log level. Defaults to 'DEBUG'.
        and_print (bool, optional): whether to print the message to the console in addition to logging. Defaults to True.
        only_print (bool, optional): whether to print the message to the console and not to log. Similar to print(). Defaults to False.
        nothing (bool, optional): Disables this function except for urgent messages. Defaults to True.
        urgent (bool, optional): When True, messages are logged and printed even if "nothing" is True. Defaults to False.
    """
    if nothing and not urgent: return
    if only_print:
        print(message)
    else:
        global using_logs
        if not using_logs :
            init_logs()

        logType = logType.lower()
        message = str(message)
        if and_print : print(message) 

        if logType == 'info' :
            logging.info( message) 
        elif logType == 'warn':
            logging.warning(message) 
        else:
            logging.debug(message) 


def read_yaml(path = 'vars.yaml'):
    """reads a given yaml file and returns its content as a dictionary

    Args:
        path (str, optional): path to yaml file. Defaults to 'vars.yaml'.

    Returns:
        Dictionary/None: returns a Dictionary containing the contents of the yaml file
        or None if the file is not found
    """
    with open(path, 'r') as stream:
        try:
            data = yaml.safe_load(stream) 
            return data
        except yaml.YAMLError as exc:
            print("error:", exc)
            return None


class cv_utils:

    def __init__():
        pass


    @staticmethod
    def read_image(path, gray = False):
        if gray:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imread(path)


    @staticmethod
    def stitch(stitcher, image1, image2, same_size):
        (status, stitched) = stitcher.stitch([image1, image2])
        if status == 0:
            log("stitched")
            if same_size:
                stitched = cv_utils.resize(stitched, image2.shape[0])
            return stitched
        else: 
            log("couldn't stitch", logType = 'warn')
            return image2

    @staticmethod
    def save_image(image, name):
        if '.' not in name: name += '.jpg'
        
        if '/' not in name and '\\' not in name :
            image_path = os.path.join(results_dir, name)
        else: image_path = name

        cv2.imwrite(image_path, image)


    @staticmethod
    def show_image(image, name = "image", loc = None, wait = 0):
        if isinstance(image, str): 
            image = cv_utils.read_image(image)

        if loc is None:
            loc = (200, 200)

        # cv2.moveWindow(name, loc[0], loc[1])
        cv2.imshow(name, image)
        cv2.waitKey(wait)
        cv2.destroyWindow(name)


    @staticmethod
    def show(images, names = None, resize = False, video_mode = False):
        assert len(images) < 9, " can not show more than 8 images"

        loc = 1
        row = 1
        for i in range(len(images)):
            if i == 4 : row, loc = 450, 1
            if names is None : name = str(i)
            else: name = names[i]
            image = images[i]
            if resize : image = cv_utils.resize(images[i], 400)
            cv2.imshow(name, image)
            #cv2.moveWindow(name, loc, row)
            loc += 420

        if video_mode :
            return

        cv2.waitKey(0)
        if names is None:
            cv2.destroyAllWindows()
        else:
            for name in names:
                cv2.destroyWindow(name)


    @staticmethod
    def gray2rgb(image):
        return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    
    @staticmethod
    def to_gray(image):
        if isinstance(image, str): 
            image = cv_utils.read_image(image)
        if cv_utils.is_gray(image): return image
   
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_binary(image, threshold = 130):
        if not cv_utils.is_gray(image): image = cv_utils.to_gray(image) 
        light = np.where(image > threshold) 
        dark = np.where(image <= threshold) 
        newimage = image.copy()
        newimage[light] = 255 
        newimage[dark] = 0
        return newimage


    #binarize the image using adaptive threshold
    @staticmethod
    def to_binary2(image):
        if not cv_utils.is_gray(image) : image = cv_utils.to_gray(image)
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY, 15, 5)
        return thresh

    
    @staticmethod
    def to_binary3(image):
        if not cv_utils.is_gray(image) : image = cv_utils.to_gray(image)
        _, thresh = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
        return thresh


    @staticmethod
    def is_gray(image):
        return len(image.shape) == 2 or image.shape[2] == 1

    @staticmethod
    def enhance(img, alpha = 2, gamma = -0.8):

        # img = unsharp_mask(img, radius=20, amount=2)
        # img = np.array((img/img.max())*255, dtype = np.uint8)

        # blur = cv2.GaussianBlur(img, (23, 23), 0)
        # img = cv2.add(img[:, :, 1], (img[:, :, 1] - blur[:, :, 1])* amount)

        gauss_mask = cv2.GaussianBlur(img, (23, 23), 10.0)
        image_sharp = cv2.addWeighted(img, alpha, gauss_mask, gamma, 0)
        return image_sharp


    @staticmethod
    def resize(image, newsize = 500):
        if isinstance(newsize, tuple) or isinstance(newsize, list):
            image = cv2.resize(image, newsize)
        else: image = cv2.resize(image, (newsize, newsize))
        return image


    @staticmethod
    def to_hsv(image):
        if isinstance(image, str): 
            image = cv_utils.read_image(image)

        assert len(image.shape)==3 and image.shape[2], \
            "can not convert to hsv. Received shape:{}".format(image.shape)
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def blur_eval(image, mask = None):

        img_fft, fft_res = cv_utils.get_fft(image, True)
        canny_res = cv_utils.auto_canny(image, mask = mask).sum()
        total_res = ((int(fft_res))*1000) + canny_res

        return fft_res, canny_res, total_res
    
    #show many images 
    @staticmethod
    def show_images(images, labels = None):
        plt.figure(figsize=(12,5))
        
        columns = len(images)
        for i, image in enumerate(images):
            if isinstance(image, str): image = cv_utils.read_image(image)
            plt.subplot(1, columns, i + 1)
            if image.shape[-1] == 3 :
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
            else: plt.imshow(image, cmap = 'gray')
            plt.axis('off')
            if labels is not None : plt.title(str(labels[i]))
        plt.show()
    
    
    @staticmethod
    def dilate(image, ksize = 5):
        if isinstance(image, str): 
            image = cv_utils.read_image(image)
        if not cv_utils.is_gray(image): image = cv_utils.to_gray(image)
        
        kernel = np.ones((ksize, ksize), dtype = np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)
        return dilated

    @staticmethod
    def close(image, ksize1 = 5, ksize2 = 5):
        if isinstance(image, str): 
            image = cv_utils.read_image(image)
        if not cv_utils.is_gray(image): image = cv_utils.to_gray(image)
        
        dilated = cv_utils.dilate(image, ksize = ksize1)
        closed = cv_utils.erode(dilated, ksize = ksize2)
        return closed


    @staticmethod
    def open(image, ksize1 = 5, ksize2 = 5):
        if isinstance(image, str): 
            image = cv_utils.read_image(image)
        if not cv_utils.is_gray(image): image = cv_utils.to_gray(image)
        
        eroded = cv_utils.erode(image, ksize = ksize1)
        opened = cv_utils.dilate(eroded, ksize = ksize2)
        return opened


    @staticmethod
    def erode(image, ksize = 5):
        if isinstance(image, str): 
            image = cv_utils.read_image(image)
        if not cv_utils.is_gray(image): image = cv_utils.to_gray(image)

        kernel = np.ones((ksize, ksize), dtype = np.uint8)
        eroded = cv2.erode(image, kernel, iterations=1)
        return eroded

    

    @staticmethod
    def median(image, ksize = 3):
        if isinstance(image, str): 
            image = cv_utils.read_image(image)

        return cv2.medianBlur(image, ksize = ksize)


    @staticmethod
    def concat_images(images):
      image = np.concatenate(images, axis=1)
      return image



    @staticmethod
    def get_fft(image, resize = False, resize_thresh = 500, with_blue_mask = False):
        
        if isinstance(image, str): 
            image = cv_utils.read_image(image)
        # if resize and (image.shape[0] > resize_thresh or image.shape[1] > resize_thresh):
            image = cv_utils.resize(image)
        
        image = cv_utils.resize(image)
        gray = cv_utils.to_gray(image)
        (h, w) = gray.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
    

        # compute the FFT to find the frequency transform, then shift
        # the zero frequency component (i.e., DC component located at
        # the top-left corner) to the center where it will be more
        # easy to analyze
        fft = np.fft.fft2(image)
        fftShift = np.fft.fftshift(fft)

        
        # zero-out the center of the FFT shift (i.e., remove low
        # frequencies), apply the inverse shift such that the DC
        # component once again becomes the top-left, and then apply
        # the inverse FFT
        fftShift[cY - 60:cY + 60, cX - 60:cX + 60] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)


        # compute the magnitude spectrum of the reconstructed image,
        # then compute the mean of the magnitude values
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        return magnitude, mean
       
        

    @staticmethod
    def auto_canny(image, sigma=0.33, resize = False, resize_thresh = 500, mask = None):
        assert mask is None or not cv_utils.is_gray(image), "can not use blue_mask with gray image"
        if isinstance(image, str): 
            image = cv_utils.read_image(image)
        if resize and (image.shape[0] > resize_thresh or image.shape[1] > resize_thresh):
            image = cv_utils.resize(image)

        if not cv_utils.is_gray(image) : image = cv_utils.to_gray(image)
       
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # compute the median of the single channel pixel intensities
        v = np.median(blurred)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(blurred, lower, upper)

        if mask is not None:
            edged = cv_utils.apply_mask(edged, mask)
        # canny = cv_utils.dilate(canny, ksize = 3)
        # return the edged image
        return edged

    @staticmethod
    def get_sperm_mask(image):
        gray = cv_utils.to_gray(image)
        sperm_loc = np.where(gray <= 120)
        other_loc = np.where(gray > 120)

        gray[sperm_loc] = 255
        gray[other_loc] = 0

        return gray


    @staticmethod
    def get_edges(image, thresh1 = 50, thresh2 = 130, resize = False, resize_thresh = 500, mask = None):
        assert mask is None or not cv_utils.is_gray(image), "can not use blue_mask with gray image"
        if isinstance(image, str): 
            image = cv_utils.read_image(image)
        if resize and (image.shape[0] > resize_thresh or image.shape[1] > resize_thresh):
            image = cv_utils.resize(image)
        

            
        if not cv_utils.is_gray(image) : image = cv_utils.to_gray(image)

        canny = cv2.Canny(image, thresh1, thresh2)

        if mask is not None:
            canny = cv_utils.apply_mask(canny, mask)
        # canny = cv_utils.dilate(canny, ksize = 3)

        return canny


    @staticmethod
    def cc(image, area_threshold = None, return_stats = False):
        if isinstance(image, str): 
            image = cv_utils.read_image(image)
        if not cv_utils.is_gray(image):
            image = cv_utils.to_gray(image)

        retval, labels, stats, centroids	=	cv2.connectedComponentsWithStats(image, connectivity=4)

        labels_to_area = [(i, stats[i, 4]) for i in range(retval)]
        labels_to_area.sort(key = lambda x: x[1], reverse=True)
        if area_threshold is None:
            if return_stats : return retval, stats, labels
            else: return labels, labels_to_area
        else:
            mask = np.zeros_like(image)
            
            for i in range(1, len(labels_to_area)): 
                label, area = labels_to_area[i]
                if area < area_threshold : continue
                mask[labels==label] = 255 
            # mask = cv_utils.dilate(mask, ksize = 5)
            return mask

    @staticmethod
    def found_color(image, mask = None, area_threshold = 500, threshold = 20):
        hsv = cv_utils.to_hsv(image)
        lower_blue = np.array([110,threshold, threshold])
        upper_blue = np.array([130,255,255])
        if mask is not None:
            hsv = cv_utils.apply_mask(hsv, mask)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return len(np.where(blue_mask == 255)[0]) > area_threshold


    @staticmethod
    def apply_mask(image, mask):
        new_image = image.copy()
        new_image[mask==0] = 0
        return new_image.astype(np.uint8)


    @staticmethod
    def anding(image, mask):
        return cv2.bitwise_and(image, mask).astype(np.uint8)
    
    @staticmethod
    def oring(image, mask):
        return cv2.bitwise_or(image, mask).astype(np.uint8)

    @staticmethod
    def xoring(image, mask):
        return cv2.bitwise_xor(image, mask).astype(np.uint8)

    @staticmethod
    def blue_mask(image, threshold = 30 ):
        hsv = cv_utils.to_hsv(image)
        lower_blue = np.array([110,threshold, threshold])
        upper_blue = np.array([200,255,255])
        
        return cv2.inRange(hsv, lower_blue, upper_blue)


    @staticmethod
    def draw_circles(image, circles):

        # Draw circles that are detected.
        if circles is not None:
            # Convert the circle parameters a, b and r to integers.
            circles = np.uint16(np.around(circles))
            for pt in circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                # Draw the circumference of the circle.
                cv2.circle(image, (a, b), r, (0, 255, 0), 2)

        return image

    @staticmethod
    def skeletonize(img):
        """ OpenCV function to return a skeletonized version of img, a Mat object"""

        #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

        img = img.copy() # don't clobber original
        skel = img.copy()

        skel[:,:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        while True:
            eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp  = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img[:,:] = eroded[:,:]
            if cv2.countNonZero(img) == 0:
                break

        return skel


class MorphologyAnalyzer:

    def __init__(self, params):
        self.params = params
        self.load_model()
    

    def load_model(self):
        self.device = torch.device('cpu')
        # self.model = torch.load(self.params['MotilityAnalyzer']['motility_model_path'], map_location = self.device)
        self.model = torch.load(self.params['MorphologyAnalyzer']['morphology_model_path'], map_location = self.device)
        self.model.eval()



    def infer(self, image, threshold = None):
        if isinstance(image, str): image = cv_utils.read_image(image)
        if threshold is None : threshold = self.params['MorphologyAnalyzer']['model_inference_threshold']

        with torch.no_grad():
            input_image = F.to_tensor(image)
            predictions = self.model([input_image.to(self.device)])
            
        masks = predictions[0]['masks'].permute(0, 2, 3, 1).mul(255).byte().cpu().numpy()
        masks = np.array((masks.astype(float)/masks.max())*255, dtype = np.uint8)
        masks = np.where(masks>threshold, 255, 0).astype(np.uint8)

        return masks
    

    def masks2mask(self, masks):
        mask = np.zeros((masks.shape[1:3]), dtype = np.uint8)
        for i in range(masks.shape[0]):
            pos = np.where(masks[i])
            mask[pos[:2]] = 255
        # mask = np.expand_dims(mask, axis = 2)
        return mask


    def smart_head_fill(self, image):
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(image, [c], -1, (255,255,255), -1)

        return image


    def get_blue_purple_mask(self, image, blue_only = False, purple_only = False ):
        
        hsv = cv_utils.to_hsv(image)
        blue_range = np.array([ [100, 45, 50], [132, 255, 255]])
        # blue_range = np.array([ [90, 45, 70], [132, 255, 255]])
        purple_range = np.array([ [132, 50, 70], [158, 255, 255]])
        mask = np.zeros(image.shape[:2], np.uint8)
        if blue_only: colors = [blue_range]
        elif purple_only: colors = [purple_range]
        else: colors = [blue_range, purple_range]

        for lower_range, upper_range in colors:
            mask = cv_utils.oring(mask, cv2.inRange(hsv, lower_range, upper_range))
        return mask
    

    def extract_parts(self, image, sperms_mask):
        sperms_mask = cv_utils.erode(sperms_mask, ksize = 3)
        sperm_labels, sperm_labels_to_area = cv_utils.cc(sperms_mask)
        sperm_labels_to_area = sperm_labels_to_area[1:] # first index is the background
        raw_sperms, heads, tails = [], [], []
        for label, area in sperm_labels_to_area:
            if area < self.params['MorphologyAnalyzer']['sperm_min_size'] : break # we use break here because areas are ordered
            raw_sperm_mask = np.where(sperm_labels==label, 255, 0).astype(np.uint8)
            raw_sperms.append(raw_sperm_mask)
        
        for raw_sperm in raw_sperms:
            head_mask = self.get_blue_purple_mask(cv_utils.apply_mask(image, raw_sperm))
            # cv_utils.show([image, head_mask])
            head_mask = cv_utils.open(head_mask, ksize1= 7, ksize2 = 7)
            head_labels, head_labels_to_area = cv_utils.cc(head_mask)
            if len(head_labels_to_area) < 2 : continue # no head is found
            head_mask = np.where(head_labels == head_labels_to_area[1][0], 255, 0).astype(np.uint8) 

            # we use continue here because a tail without head might have area (and order) bigger than a tail with head.
            head_size = (head_mask.sum()//255)
            if head_size < self.params['MorphologyAnalyzer']['head_min_size'] : continue 
            heads.append(head_mask) 

            raw_sperm = cv_utils.median(raw_sperm, ksize = 5)
            tail_mask = self.skeletonize(raw_sperm)
            tail_mask = cv_utils.anding(tail_mask, ~head_mask)
            tail_mask = cv_utils.dilate(tail_mask, ksize = 3) 
            tail_labels, tail_labels_to_area = cv_utils.cc(tail_mask) 
            tail_mask = np.where(tail_labels == tail_labels_to_area[1][0], 255, 0).astype(np.uint8)
            # for i in range(2, len(tail_labels_to_area)):
            #     if tail_labels_to_area[i][1]*10 < tail_labels_to_area[1][1] : break 
            #     tail_mask += np.where(tail_labels == tail_labels_to_area[i][0], 255, 0).astype(np.uint8)
            tails.append(tail_mask)

        return heads, tails


    def analyze_head(self, image, head_mask):
        # head_mask = self.smart_head_fill(head_mask) 
        masked_image = cv_utils.apply_mask(image, head_mask)
        blue_mask = self.get_blue_purple_mask(masked_image, blue_only = True)
        purple_mask = self.get_blue_purple_mask(masked_image, purple_only = True)
        return blue_mask, purple_mask


    def draw_things(self, blue_head, purple_head, tail):
        colored_image = np.zeros((blue_head.shape[0], blue_head.shape[1], 3), dtype = np.uint8)

        colored_image = np.where(blue_head[..., None], np.array([255, 0, 0]), np.array([0, 0, 0]))
        colored_image += np.where(purple_head[..., None], [128, 0, 128], [0, 0, 0])
        colored_image += np.where(tail[..., None], [0, 255, 0], [0, 0, 0])
        return colored_image.astype(np.uint8)

    def skeletonize(self, image, mask = None):
        new_image = np.array(image/image.max())
        # skeleton = cv_utils.skeletonize(new_image).astype(np.uint8)
        skeleton = morphology.skeletonize(new_image).astype(np.uint8)
        skeleton*= 255
        return skeleton

    def analyze(self, img):
        image = img.copy()
        image = cv_utils.enhance(image)
        masks = self.infer(image)
        mask = self.masks2mask(masks)

        heads, tails = self.extract_parts(image, mask)
        
        blue_heads, purple_heads = [], []
        for head in heads:
            blue_head, purple_head = self.analyze_head(image, head)
            blue_heads.append(blue_head)
            purple_heads.append(purple_head)
        
        colored_image = np.zeros_like(image, dtype = np.uint8)
        for blue_head, purple_head, tail in zip(blue_heads, purple_heads, tails):
            colored_image += self.draw_things(blue_head, purple_head, tail)

        return colored_image


def main():
    image_path = os.path.join(morphology_images_dir, os.listdir(morphology_images_dir)[5])
    image = cv_utils.read_image(image_path)
    

    params = read_yaml()
    m = MorphologyAnalyzer(params)

    res = m.analyze(image) 
    cv_utils.show([image, res])


if __name__ == '__main__':
    main()

