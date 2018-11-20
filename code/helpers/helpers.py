from PIL import Image, ImageOps, ImageFilter
import glob
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------

def load_data():
    """Loads and returns images and labels."""
    
    # Creates a list of file names in the data directory
    filelist = glob.glob("../data/Images/*.tif")
    
    # Loads all data images in a list
    data = [Image.open(fname) for fname in filelist]
    
    # Creates a list of file names in the labels directory
    filelist = glob.glob("../data/Labels/*.tif")
    
    # Loads all labels images in a list
    labels = [Image.open(fname) for fname in filelist]

    return data, labels

# -----------------------------------------------------------------------------

def save_data(data, labels, tr):
    """Save images and labels."""
    
    direc_d = []
    direc_l = []
    
    # Selects path for data and labels in the training folder
    if tr == True:
        direc_d = "../randomized_data/train/data_{b}.tiff"
        direc_l = "../randomized_data/train/data_{b}_mask.tiff"
        
    # Selects path for data and labels in the testing folder
    else:
        direc_d = "../randomized_data/test/data_{b}.tiff"
        direc_l = "../randomized_data/test/data_{b}_mask.tiff"
    
    # Saves data and labels in the right folder
    for i in range(len(data)):
        data[i].save(direc_d.format(b=i))
        labels[i].save(direc_l.format(b=i))
        
    return None

# -----------------------------------------------------------------------------

def split_data(X, y, ratio=0.8, seed=1):
    """The split_data function will shuffle data randomly as well as return
    a split data set that are individual for training and testing purposes.
    The input X is a list of images. The input y is a list of binary images
    with each image corresponding to the label of the corresponding sample in X. 
    The ratio variable is a float, default 0.8, that sets the train set fraction of
    the entire dataset to 0.8 and keeps the other part for test set"""
        
    # Set seed
    np.random.seed(seed)

    # Perform shuffling
    idx_shuffled = np.random.permutation(len(y))
    
    # Return shuffled X and y
    X_shuff = [X[i] for i in idx_shuffled]
    y_shuff = [y[i] for i in idx_shuffled]

    # Cut the data set into train and test
    train_num = round(len(y) * ratio)
    X_train = X_shuff[:train_num]
    y_train = y_shuff[:train_num]
    X_test = X_shuff[train_num:]
    y_test = y_shuff[train_num:]

    return X_train, y_train, X_test, y_test, idx_shuffled

# -----------------------------------------------------------------------------

def smoothing_edge (label):
    """Smoothens the edges of the labels."""
    
    tmp = label
    
    # Labels are inverted (background = white, signal = black),
    # this step inverts that.
    tmp = ImageOps.invert(tmp)
    
    # Erosion of the labelled image
    tmp = tmp.filter(ImageFilter.MinFilter(3))
    
    # Median filter to make edges a bit smoother
    tmp = tmp.filter(ImageFilter.MedianFilter(5))
    tmp = tmp.filter(ImageFilter.MedianFilter(3))
    
    return tmp

# -----------------------------------------------------------------------------

def gradient_detector(data_img):
    """Detects the direction of the gradient image."""
    
    row, col = False, False
    
    # Image size
    n = len(data_img)
    half_n = np.int(n/2 - 1)
    
    # Construction of X data for regression
    X_reg = np.linspace(0,n-1,n)
    X_reg = X_reg.reshape(-1,1)
    
    # Construction of y data for regression at the center of the image
    # along rows
    y_reg = data_img[half_n]
    
    # Regression fit
    reg = LinearRegression().fit(X_reg,y_reg)
    
    # Storing the slope of the regression
    row_coef = reg.coef_
        
    # Construction of y data for regression at the center of the image
    # along columns
    y_reg = data_img.T[half_n]
    
    # Regression fit
    reg = LinearRegression().fit(X_reg,y_reg)
    
    # Storing the slope of the regression
    col_coef = reg.coef_
    
    # This steps finds which direction the gradient will be (along rows or columns)
    if np.abs(row_coef) > np.abs(col_coef):
        row = True
    
    else:
        col = True
    
    return row, col

# -----------------------------------------------------------------------------

def gradient_generator(data_img, row = True, col = False):
    """Generates an image with a gradient in the horizontal or
    vertical direction."""
    
    # Image size
    n = len(data_img)
    
    # Initialization of gradient
    grad = np.zeros((n,n))
    
    # Initialization of min and max values for gradient
    data_min = np.zeros((n,1))
    data_max = np.zeros((n,1))
    
    # Generates gradient along rows
    if row:
        
        # Construction of X data for regression
        X_reg = np.linspace(0,n-1,n)
        X_reg = X_reg.reshape(-1,1)
        
        # This for loop computes a regression of the pixel values
        # along rows in order to find predicted min and max values
        # in order to construct a gradient for each row afterwards
        for i in range(n):
            
            # Construction of y data for regression
            y_reg = data_img[i]
            
            # Regression fit
            reg = LinearRegression().fit(X_reg,y_reg)
            data_min[i] = reg.intercept_
            data_max[i] = reg.coef_*(n-1) + reg.intercept_
        
        # Take average min and max values
        data_min = np.abs(np.mean(data_min))
        data_max = np.abs(np.mean(data_max))
        
        # These for loops construct gradients along rows
        # using average min and max values from the regressions
        
        for i in range(n):
            for j in range(n):
                grad[i][j] = ((data_max-data_min)/(n-1)) * j + data_min

    # Generates gradient along columns
    if col:
        
        # Construction of X data for regression
        X_reg = np.linspace(0,n-1,n)
        X_reg = X_reg.reshape(-1,1)
        
        # This for loop computes a regression of the pixel values
        # along rows in order to find predicted min and max values
        # in order to construct a gradient for each column afterwards
        for i in range(n):
            
            # Construction of y data for regression
            y_reg = data_img.T[i]
            
            # Regression fit
            reg = LinearRegression().fit(X_reg,y_reg)
            data_min[i] = reg.intercept_
            data_max[i] = reg.coef_*(n-1) + reg.intercept_
        
        # Take average min and max values
        data_min = np.mean(data_min)
        data_max = np.mean(data_max)
        
        # These for loops construct gradients along columns
        # using average min and max values from the regressions
        for i in range(n):
            for j in range(n):
                grad[j][i] = ((data_max-data_min)/(n-1)) * j + data_min
        
    return grad

# -----------------------------------------------------------------------------
    
def apply_gradient(data_img, gradient):
    """Applies the gradient to the image."""
    
    # Average value of the gradient image
    average = np.mean(data_img)
    
    # Computing our corrected image
    output = data_img - gradient + average
    
    # Correction of pixel value
    output[output >= 255] = 255
    output[output <= 0] = 0
    
    return output

# -----------------------------------------------------------------------------

def illum_correction(image):
    """"This function corrects a common illumination issue which is when
    an image has darker regions and lighter regions due to issues related
    to the system of acquisition. Illumination correction is perform
    along rows or columns."""
    
    # Convert image to array
    data_img = np.array(image, dtype=float)
    
    # Detect gradient
    row, col = gradient_detector(data_img)
    
    # If no gradient is detectedm no correction is performed
    if (row == False) and (col == False):
        return image
    
    # Create gradient image
    grad = gradient_generator(data_img, row, col)
    
    # Apply gradient
    output = apply_gradient(data_img, grad)
    
    # Converting the output values into integers
    # for the conversion into an image to work
    output = output.astype('uint8')
    
    # Convert the array into an image
    output_img = Image.fromarray(output,'L')
    
    return output_img

# -----------------------------------------------------------------------------

def normalize(image):
    """Normalizes luminance to (mean,std) = (0,1)
    and applies a [1%, 99%] contrast stretch."""
    
    # Convert image to array
    data_img = np.array(image, dtype=float)
    
    # Normalizes image to (mean,std) = (0,1)
    data_img = data_img/255
    data_img -= data_img.mean()
    
    
    if data_img.std() != 0:
        data_img /= data_img.std()
    
    # Construction of contrast stretch and apply
    scale = np.max([np.abs(np.percentile(data_img, 1.0)),
                np.abs(np.percentile(data_img, 99.0))])
    data_img = data_img / scale
    data_img = np.clip(data_img, -1.0, 1.0)
    
    # Correction
    data_img = (data_img + 1.0) / 2.0
    
    # Convert into uint8 for correct image conversion
    data_img = (data_img * 255 + 0.5).astype('uint8')
    
    # Conversion into image
    output = Image.fromarray(data_img)
    
    return output

# -----------------------------------------------------------------------------

def cropping(data, labels):
    """Divides the images in 4 images"""
    
    tmp = []
    l_tmp = []
    
    length = np.array(data[0], dtype=float)
    
    n = len(length)
    half_n = np.int(n/2)
    
    data_copy = data.copy()
    labels_copy = labels.copy()
    
    print("Performing cropping")
    
    # First quadrant cropping
    print("First quadrant cropping")
    for i in range(len(data_copy)):
        tmp.append(data_copy[i].crop((0,0,half_n,half_n)))
        l_tmp.append(labels_copy[i].crop((0,0,half_n,half_n)))
        
    data_copy = data.copy()
    labels_copy = labels.copy()
    
    print("Second quadrant cropping")
    # Second quadrant cropping
    for i in range(len(data_copy)):
        tmp.append(data_copy[i].crop((0,half_n,half_n,n)))
        l_tmp.append(labels_copy[i].crop((0,half_n,half_n,n)))
        
    data_copy = data.copy()
    labels_copy = labels.copy()
    
    print("Third quadrant cropping")
    # Third quadrant cropping
    for i in range(len(data_copy)):
        tmp.append(data_copy[i].crop((half_n,0,n,half_n)))
        l_tmp.append(labels_copy[i].crop((half_n,0,n,half_n)))
    
    data_copy = data.copy()
    labels_copy = labels.copy()
        
    print("Fourth quadrant cropping")
    # Fourth quadrant cropping
    for i in range(len(data_copy)):
        tmp.append(data_copy[i].crop((half_n,half_n,n,n)))
        l_tmp.append(labels_copy[i].crop((half_n,half_n,n,n)))
    
    print("Cropping - Done")
    
    return tmp, l_tmp

# -----------------------------------------------------------------------------

def rotate(data, labels):
    """Divides the images in 4 images"""
    
    tmp = data.copy()
    l_tmp = labels.copy()
    print("Performing rotations")
    
    data_copy = data.copy()
    labels_copy = labels.copy()
    
    # First rotation 90°
    print("90° Rotation")
    for i in range(len(data_copy)):
        tmp.append(data_copy[i].transpose(Image.ROTATE_90))
        l_tmp.append(labels_copy[i].transpose(Image.ROTATE_90))
    
    data_copy = data.copy()
    labels_copy = labels.copy()
    
    print("180° Rotation")
    # First rotation 180°
    for i in range(len(data_copy)):
        tmp.append(data_copy[i].transpose(Image.ROTATE_180))
        l_tmp.append(labels_copy[i].transpose(Image.ROTATE_180))
    
    data_copy = data.copy()
    labels_copy = labels.copy()
    
    print("270° Rotation")
    # First rotation 270°
    for i in range(len(data_copy)):
        tmp.append(data_copy[i].transpose(Image.ROTATE_270))
        l_tmp.append(labels_copy[i].transpose(Image.ROTATE_270))
    
    print("Rotations - Done")
    
    return tmp, l_tmp

# -----------------------------------------------------------------------------

def preprocessing(data, labels, label_smooth = True, illum_cor = True, norm = True, crop = True, rotation = True):
    """This function applies various preprocessing techniques
    such as background normalization, edge-smoothing, etc..."""
    
    # Copy of data and labels
    data_ = data
    labels_ = labels
    
    # Edge-smoothing of ground-truth labels using a median filter of size 5x5
    if label_smooth:
        print("Applying edge-smoothing to the labels")
        for i in range(len(labels_)):
            labels_[i] = smoothing_edge(labels_[i])
        print("Edge-smoothing of labels - Done")
            
    # Illumination correction
    if illum_cor:
        print("Performing illumination correction on images")
        for i in range(len(data_)):
            data_[i] = illum_correction(data_[i])

        print("Illumination correction - Done")
        
    # Normalization
    if norm:
        print("Normalizing")
        for i in range(len(data_)):
            data_[i] = normalize(data_[i])
        print("Normalization - Done")
        
    # Divide images in 4
    if crop:
        data_, labels_ = cropping(data_, labels_)
       
    # Rotate images in 4 directions
    if rotation:    
        data_, labels_ = rotate(data_, labels_)
        
    return data_, labels_