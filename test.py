# Face Identification Algorithm using MTCNN for face detection, FaceNet for face feature extraction, and SVM for Classification 
from sklearn.svm import SVC
from numpy import load
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
from PIL import Image
from mtcnn.mtcnn import MTCNN
import numpy as np

# extract a single face from a given photograph using MTCNN
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # check if face was detected
    if results:
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    else:
        return None

# get the face embedding for one face using FaceNet
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

# load the face embeddings dataset
data = load('./Embeddings_Dataset/Embeddings.npz')
trainX, trainy = data['arr_0'], data['arr_1']
print('Loaded train embeddings: ', trainX.shape, trainy.shape)

# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded FaceNet model')

# extract the face embedding for a new image
image = extract_face('./Data/Team_Validation_Dataset/Ahmad/2.jpeg')
embedding = get_embedding(model, image)
testX = asarray([embedding])
print('Loaded test embedding: ', testX.shape)
# 7.3 12 ravilya unknown
# fit SVM model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# compute distances between embeddings of test image and all training images
distances = np.linalg.norm(trainX - testX, axis=1)

# set threshold for the minimum distance allowed
threshold = 0

# defining the authorized users
authorized_users = ['Ravilya', 'Ahmad', 'Milena']

# predict the class of the new image
if np.min(distances) <= threshold:
    # get the predicted class and its corresponding probability
    yhat_test = model.predict(testX)[0]
    proba = model.predict_proba(testX)[0]
    
    if yhat_test in authorized_users:
        print('Prediction Result: ', yhat_test, ' - Authorized')
        print('Confidence Rate:', proba[authorized_users.index(yhat_test)])
    else:
         print('Unauthorized')
    
   
else:
    # consider the test image an unknown face
    print('Unknown Face')
