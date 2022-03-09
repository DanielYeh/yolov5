import os
import cv2
import time
import glob
import shutil
import random
import imutils
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Generator(tf.keras.utils.Sequence):

    def __init__(self, DATASET_PATH, CNT_PATH, LABELS_PATH, BATCH_SIZE=32, shuffle_images=False, image_min_side=24, n_class = 2, image_size = (320,320)):
        """ Initialize Generator object.

        Args
            DATASET_PATH           : Path to folder containing individual folders named by their class names
            CNT_PATH               : Path to folder containing contour images
            LABELS_PATH            : Path to folder containing contour labels
            BATCH_SIZE             : The size of the batches to generate.
            shuffle_images         : If True, shuffles the images read from the DATASET_PATH
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
        """

        self.batch_size = BATCH_SIZE
        self.shuffle_images = shuffle_images
        self.image_min_side = image_min_side
        # self.load_image_paths_labels(DATASET_PATH, LABELS_PATH) # Ewan's
        # self.create_image_groups()
        self.create_img_arr(DATASET_PATH, CNT_PATH, LABELS_PATH)
        self.n_class = n_class
        self.image_size = image_size
        
    def process_image(self, image):
        rgb_mean = np.array([0.485, 0.456, 0.406])
        rgb_std = np.array([0.229, 0.224, 0.225])

        # randomly shift gamma
        gamma = random.uniform(0.8, 1.2)
        image = image.copy() ** gamma
        image = np.clip(image, 0, 255)

        # randomly shift brightness
        brightness = random.uniform(0.5, 1.2)
        image = image.copy() * brightness
        image = np.clip(image, 0, 255)
        
        # image transformation here
        # image = (image / 255. - rgb_mean) / rgb_std
        fImg = image.astype(np.float32)/255.0
        hlsImg = cv2.cvtColor(fImg, cv2.COLOR_RGB2HLS)
        saturation = random.uniform(0.5, 150)
        hlsImg[:,:,2] = (1 + saturation / 100.0) * hlsImg[:, :, 2]
        hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 0.8
        resultIMG = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
        resultIMG = (resultIMG * 255).astype(np.uint8)
    
        # label = open(label_path).readlines()
        # label = [np.array(line.rstrip().split(" ")) for line in label]
        # label = np.array(label, dtype=np.int)
        # label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
        # label = label.astype(np.int)
    
        return resultIMG

    def load_image_paths_labels(self, DATASET_PATH, LABELS_PATH):
        
        classes = os.listdir(DATASET_PATH)
        # lb = preprocessing.LabelBinarizer()
        # lb.fit(classes)

        self.image_paths = []
        self.image_labels = []
        # for class_name in classes:
        #     class_path = os.path.join(DATASET_PATH, class_name)
        for image_file_name in os.listdir(DATASET_PATH):
            self.image_paths.append(os.path.join(DATASET_PATH, image_file_name))
            self.image_labels.append(os.path.join(LABELS_PATH, image_file_name.split(".")[0]+".txt"))

        # self.image_labels = np.array(self.image_labels, dtype='float32')
        
        # assert len(self.image_paths) == len(self.image_labels)

    def create_img_arr(self, DATASET_PATH, CNT_PATH, LABELS_PATH):
        self.image_paths = []
        self.cnt_paths = []
        self.image_labels = []

        for image_file_name in os.listdir(DATASET_PATH):
            self.image_paths.append(os.path.join(DATASET_PATH, image_file_name))
            self.cnt_paths.append(os.path.join(CNT_PATH, image_file_name))
            self.image_labels.append(os.path.join(LABELS_PATH, image_file_name.split(".")[0]+".txt"))

        if self.shuffle_images:
            # Randomly shuffle dataset
            seed = 4321
            np.random.seed(seed)
            np.random.shuffle(self.image_paths)
            np.random.seed(seed)
            np.random.shuffle(self.cnt_paths)
            np.random.seed(seed)
            np.random.shuffle(self.image_labels)

        # Divide image_paths and image_labels into groups of BATCH_SIZE
        self.image_groups = [[self.image_paths[x % len(self.image_paths)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.image_paths), self.batch_size)]
        self.cnt_groups = [[self.cnt_paths[x % len(self.cnt_paths)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.cnt_paths), self.batch_size)]
        self.label_groups = [[self.image_labels[x % len(self.image_labels)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.image_labels), self.batch_size)]

    def create_image_groups(self):
        if self.shuffle_images:
            # Randomly shuffle dataset
            seed = 4321
            np.random.seed(seed)
            np.random.shuffle(self.image_paths)
            np.random.seed(seed)
            np.random.shuffle(self.image_labels)

        # Divide image_paths and image_labels into groups of BATCH_SIZE
        self.image_groups = [[self.image_paths[x % len(self.image_paths)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.image_paths), self.batch_size)]
        self.label_groups = [[self.image_labels[x % len(self.image_labels)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.image_labels), self.batch_size)]

    def resize_image(self, img, min_side_len):

        h, w, c = img.shape

        # limit the min side maintaining the aspect ratio
        if min(h, w) < min_side_len:
            im_scale = float(min_side_len) / h if h < w else float(min_side_len) / w
        else:
            im_scale = 1.

        new_h = int(h * im_scale)
        new_w = int(w * im_scale)

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return re_im, new_h / h, new_w / w

    def load_images(self, image_group, cnt_group, label_group):

        images = []
        contours = []
        labels = []
        for image_path in image_group:
            img = cv2.imread(image_path)
            img_shape = len(img.shape)
            if img_shape == 2:
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            elif img_shape == 4:
                img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
            elif img_shape == 3:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # img, rh, rw = self.resize_image(img, self.image_min_side)
            # img = cv2.resize(img,self.image_size,interpolation=cv2.INTER_AREA)
            # if random.uniform(0.0, 1.0) > 0.5:
            img = self.process_image(img)
            images.append(img)
        for cnt_path in cnt_group:
            img = cv2.imread(cnt_path)
            contours.append(img)
        for label_path in label_group:
            with open(label_path) as file:
                # array2d = [[float(digit) for digit in line.split()] for line in file]
                # array2d = cv2.resize(np.array(array2d),self.image_size,interpolation=cv2.INTER_AREA)
                # array2d = np.ceil(array2d)
                # labels.append(self.expand_label_info(array2d))
                # labels.append(cv2.resize(np.array(array2d),self.image_size,interpolation=cv2.INTER_AREA))
                label = file.read()
                labels.append(label)
                
        return images, contours, labels

    def expand_label_info(self,label):
        label_out = np.zeros(shape = [self.image_size[0], self.image_size[1], self.n_class])
        for c in range(self.n_class):
            label_out[:, :, c] = (label == c).astype(int)
        label_out = np.reshape(label_out, (-1, self.n_class))
        return label_out

    def construct_image_batch(self, image_group, contour_group, label_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        max_shape_contour = tuple(max(image.shape[x] for image in contour_group) for x in range(3))
        # max_shape_label = tuple(max(label.shape for label in label_group))
        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype='float32') #(batch, h,w,c)
        contour_batch = np.zeros((self.batch_size,) + max_shape_contour, dtype='float32')
        # label_batch = np.zeros((self.batch_size,) + max_shape_label, dtype='float32')
        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image
            contour_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = contour_group[image_index]
            # label_batch[image_index, :label_group[image_index].shape[0], :label_group[image_index].shape[1]] = label_group[image_index]

        return image_batch, contour_batch, label_group

    def normalize(self, x):
        """
        Normalize a list of sample image data in the range of 0 to 1
        : x: List of image data.  The image shape is (320, 320, 3)
        : return: Numpy array of normalized data
        """
        return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    
    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.image_groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        image_group = self.image_groups[index]
        contour_group = self.cnt_groups[index]
        label_group = self.label_groups[index]
        images, contours, labels = self.load_images(image_group, contour_group, label_group)
        image_batch, contour_batch, label_batch = self.construct_image_batch(images, contours, labels)

        # image_batch = self.normalize(image_batch)
        # label_batch = self.normalize(label_batch)

        return image_batch, contour_batch, label_batch

        # return np.array(image_batch), np.array(label_batch)

# find the contour of croped contour image and write it to txt file
def find_contour(cnt_img, path, filename, imgsz, cnt):
    gray = cv2.cvtColor(cnt_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,50,255,0)
    try:
        if imutils.is_cv2() or imutils.is_cv4():
            contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        elif imutils.is_cv3():
            img, contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    except:
        print("Error: Could not find any contours in image")
    
    name_num = ''.join([i for i in filename if i.isdigit()])
    
    classname = 'remap_dummy_' + str(int(name_num) % 4) + 'p'
    if classname == 'remap_dummy_0p':
        classname = 'remap_dummy_4p'

    # return yolo format txt file (x, y, width, height) in normalized
    with open(path + '/labels/' + 'blemish_' + str(cnt)+ '.txt', 'w') as f:
        try:
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                f.write(str(int(name_num)-1) + ' ' + str(x/imgsz) + ' ' + str(y/imgsz) 
                        + ' ' + str(w/imgsz) + ' ' + str(h/imgsz) + '\n')
        except:
            print('Fail in contour')
    f.close()

# find the main ROI of image
def find_roi(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,50,255,0)
	if imutils.is_cv2() or imutils.is_cv4():
		contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	elif imutils.is_cv3():
		img, contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	approx = cv2.approxPolyDP(contours[0],50,True)
	mask = cv2.drawContours(image*0,approx, -1, (255), 15)
    
    # if our approximated contour has four points, then we
    # can assume that we have found our piece of paper
	if len(approx) == 4:
		return approx
	
	area=[]
	for cnt in contours:
		area.append(cv2.contourArea(cnt))
	contour_n=np.argmax(np.array(area))
	roi = cv2.boundingRect(contours[contour_n]) #x,y,w,h
	
	return roi

def sliding_window(image, contour_img, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]], 
                   contour_img[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, contour_img, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image, contour_img

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        contour_img = imutils.resize(contour_img, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image, contour_img

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=r'.\data\images\remap_dummy_4p_l.png', help="Path to the image")
    ap.add_argument("-c", "--contour", default=r'.\data\images\blm_contour_l_remap_dummy_4p.jpg', help="Path to the contour image")
    ap.add_argument("-p", "--path", default=r"../datasets/blemish", help="Path to the dataset folder")
    args = vars(ap.parse_args())
    path = args["path"]

    # define the window width and height
    (winW, winH) = (320, 320)
    COUNT = 0
    
    for file in glob.glob(path + '/origin_img/*.png'):
        filename = os.path.splitext(os.path.split(file)[1])[0]
        image = cv2.imread(path + "/origin_img/" + filename + ".png")
        contourImg = cv2.imread(path + "/origin_cnt/" + filename + ".png")

        # find the main ROI of image and check if it is valid
        roi = find_roi(image)
        if len(roi) == 4:
            if issubclass(type(roi), tuple):
                image = image[roi[1]:roi[3], roi[0]:roi[2]]
                contourImg = contourImg[roi[1]:roi[3], roi[0]:roi[2]]
            else:
                image = image[roi[1,0,1]:roi[3,0,1], roi[1,0,0]:roi[3,0,0]]
                contourImg = contourImg[roi[1,0,1]:roi[3,0,1], roi[1,0,0]:roi[3,0,0]]
        else:
            print('ROI not found')
            exit()

        # loop over the image pyramid
        for resized, contour in pyramid(image, contourImg, scale=1.5):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window, cnt) in sliding_window(resized, contour, stepSize=int(winW*0.75), 
                                                    windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                
                # find contour and save image
                find_contour(cnt, path, filename, winW, COUNT)
                cv2.imwrite(path + "/tmp/images/blemish_" + str(COUNT) + ".png", window)
                cv2.imwrite(path + "/tmp/contours/blemish_" + str(COUNT) + ".png", cnt)
                COUNT += 1

        # Generate augmentation object
        train_generator = Generator(path + "/tmp/images/", path + "/tmp/contours/", path + "/labels/",
                                    BATCH_SIZE=10, n_class=2, image_size=(winW, winH))

        print(len(train_generator))
        
        for i in range(0, len(train_generator)):
            image_batch, cnt_batch, label_batch = train_generator.__getitem__(i)
            print("Processing {}, batch {} / {}".format(filename, i, len(train_generator)))
            print(image_batch.shape)
            print(cnt_batch.shape)
            print(len(label_batch))

            for i in range(0, len(image_batch)):
                # find_contour(cnt_batch[i], path, filename, winW, COUNT)
                cv2.imwrite(path + "/images/blemish_" + str(i + COUNT) + ".png", image_batch[i])
                cv2.imwrite(path + "/contours/blemish_" + str(i + COUNT) + ".png", cnt_batch[i])
                with open(path + "/labels/blemish_" + str(i + COUNT) + ".txt", 'w') as f:
                    try:
                        f.write(label_batch[i])
                    except:
                        print("Error writing label")
                    f.close()
            COUNT += len(image_batch)
        
        # Move images from tmp folder to dataset folder
        for file in glob.glob(path + '/tmp/images/*.png'):
            shutil.move(file, path + "/images/")
        for file in glob.glob(path + '/tmp/contours/*.png'):
            shutil.move(file, path + "/contours/")
        
        print("Done: {}".format(filename))
