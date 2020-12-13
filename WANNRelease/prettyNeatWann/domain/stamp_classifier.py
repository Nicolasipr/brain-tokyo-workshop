import cv2
import gym
import numpy as np
from alerce.core import Alerce
from gym import spaces
from gym.utils import seeding

alerce_client = Alerce()

LABELS = {
"AGN": 0, "SN":1, "VS":2, "Asteroid":3, "Bogus":4
}

class StampClasiffierEnv(gym.Env):
    """Classification as an unsupervised OpenAI Gym RL problem.
    Includes scikit-learn digits dataset, MNIST dataset
    """

    def __init__(self, trainSet, target):
        """
        Data set is a tuple of
        [0] input data: [nSamples x nInputs]
        [1] labels:     [nSamples x 1]

        Example data sets are given at the end of this file
        """

        self.t = 0  # Current batch number
        self.t_limit = 0  # Number of batches if you need them
        self.batch = 1000  # Number of images per batch
        self.seed()
        self.viewer = None

        self.trainSet = trainSet
        self.target = target

        nInputs = np.shape(trainSet)[1]
        high = np.array([1.0] * nInputs)
        self.action_space = spaces.Box(np.array(0, dtype=np.float32), \
                                       np.array(1, dtype=np.float32))
        self.observation_space = spaces.Box(np.array(0, dtype=np.float32), \
                                            np.array(1, dtype=np.float32))

        self.state = None
        self.trainOrder = None
        self.currIndx = None

    def seed(self, seed=None):
        ''' Randomly select from training set'''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        ''' Initialize State'''
        # print('Lucky number', np.random.randint(10)) # same randomness?
        self.trainOrder = np.random.permutation(len(self.target))
        self.t = 0  # timestep
        self.currIndx = self.trainOrder[self.t:self.t + self.batch]
        self.state = self.trainSet[self.currIndx, :]
        return self.state

    def step(self, action):
        '''
        Judge Classification, increment to next batch
        action - [batch x output] - softmax output
        '''
        y = self.target[self.currIndx]
        m = y.shape[0]
        log_likelihood = -np.log(action[range(m), y])
        loss = np.sum(log_likelihood) / m
        reward = -loss

        if self.t_limit > 0:  # We are doing batches
            reward *= (1 / self.t_limit)  # average
            self.t += 1
            done = False
            if self.t >= self.t_limit:
                done = True
            self.currIndx = self.trainOrder[(self.t * self.batch): \
                                            (self.t * self.batch + self.batch)]

            self.state = self.trainSet[self.currIndx, :]
        else:
            done = True

        obs = self.state
        return obs, reward, done, {}


# -- Data Sets ----------------------------------------------------------- -- #

flatten = lambda t: [item for sublist in t for item in sublist]


def get_objects_per_class(classearly="SN", pclassearly=0.5, n_objects=100):
    objects = alerce_client.query_objects(
        classifier="stamp_classifier",
        class_name=classearly,
        probability=pclassearly,
        count=False,
        page_size=n_objects,
        format='pandas')
    objects.head()
    objects.set_index("oid", inplace=True)
    objects.sort_values(by="ndet", inplace=True, ascending=False)
    return objects


def fetch_sample_date(n_objects=500):
    early_classes = ["AGN", "SN", "VS", "Asteroid", "Bogus"]  # Class identifiers to query objects
    sample_data = []
    labels = []
    data = {}

    # Fetches each class into a pandas object
    for i, cl in enumerate(early_classes):
        data[cl] = get_objects_per_class(classearly=cl, n_objects=n_objects)

    for item in data:
        sample_data.append(data[item].index)  # selects all oids of the current class
        # labels.append(list(range(item, len(labels))))
        labels.append(data[item]['class'].values)  # selects all classes of the current class

    flat_samples = flatten(sample_data)  # flatten the data
    flat_labels = flatten(labels)
    parsed_labels = []
    for i in flat_labels:
        parsed_labels.append(LABELS[i])
    parsed_labels = np.array(parsed_labels)
    return flat_samples, parsed_labels


def get_stamps(samples):
    stamps_samples = []
    for oid in samples:
        try:
            stamps = alerce_client.get_stamps(oid)
            stamps_samples.append(stamps)
        except Exception as e:
            print('unable to fetch: ', oid)
    return stamps_samples


def format_stamps(stamps):
    """
    Converts 63x63 stamps to 16x16
    [samples x pixels]  ([N X 784])
    """
    stamps_data = []
    for stamp in stamps:
        science, template, difference = [i.data for i in stamp]
        stamps_data.append(science)
    stamps_data = np.array(stamps_data, dtype=np.float32)
    stamps_data = preprocess(stamps_data,(16,16), unskew=False)
    stamps_data = stamps_data.reshape(-1, (256))
    return stamps_data


def stamps_wrapper():

    samples, targets = fetch_sample_date(n_objects=2500)
    samples_stamps = get_stamps(samples=samples)
    formatted = format_stamps(samples_stamps)
    stamp_formatted = np.array(formatted)
    return stamp_formatted, targets



def preprocess(img, size, patchCorner=(0, 0), patchDim=None, unskew=True):
    """
    Resizes, crops, and unskewes images

    """
    if patchDim == None: patchDim = size
    nImg = np.shape(img)[0]
    procImg = np.empty((nImg, size[0], size[1]))

    # Unskew and Resize
    if unskew == True:
        for i in range(nImg):
            procImg[i, :, :] = deskew(cv2.resize(img[i, :, :], size), size)

    # Crop
    cropImg = np.empty((nImg, patchDim[0], patchDim[1]))
    for i in range(nImg):
        cropImg[i, :, :] = procImg[i, patchCorner[0]:patchCorner[0] + patchDim[0], \
                           patchCorner[1]:patchCorner[1] + patchDim[1]]
    procImg = cropImg
    return procImg


def deskew(image, image_shape, negated=True):
    """
    This method deskwes an image using moments
    :param image: a numpy nd array input image
    :param image_shape: a tuple denoting the image`s shape
    :param negated: a boolean flag telling whether the input image is negated

    :returns: a numpy nd array deskewd image

    source: https://github.com/vsvinayak/mnist-helper
    """
    # negate the image
    if not negated:
        image = 255 - image
    # calculate the moments of the image
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    # caclulating the skew
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * image_shape[0] * skew], [0, 1, 0]])
    img = cv2.warpAffine(image, M, image_shape, \
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
