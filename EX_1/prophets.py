import numpy as np
import pandas as pd
from tqdm import tqdm
# a=np.array([1,2,3,4,5,6,7,8,9,10],dtype='int16')
# b=np.array([[1,2,3,4,5,6,7,8,9,10],
#             [1,2,3,4,5,6,7,8,9,10]])
# dim_b=b.ndim #get dimension
# shap_b=b.shape #get shape // (2,10)
# type_b=b.dtype #get type
# total_size=b.nbytes #get total size
# b[:,0] #get first column
# b[0,:] #get first row
# #get diagonal elements
# np.diag(b)
# np.random.choice(a,5) #randomly choose 5 elements from a
# np.random.uniform(0,1,10) #randomly choose 10 elements from uniform distribution
# np.random.int(0,10,10) #randomly choose 10 elements from int distribution
def sample_prophets(k, min_p, max_p):
    """
    Samples a set of k prophets
    :param k: number of prophets
    :param min_p: minimum probability
    :param max_p: maximum probability
    :return: list of prophets
    """
    prob_list=np.random.uniform(min_p,max_p,k)
    prophets=[]
    for i in range(prob_list.shape[0]):
        prophets.append(Prophet(prob_list[i]))
    return prophets

class Prophet:

    def __init__(self, err_prob):
        """
        Initializes the Prophet model
        :param err_prob: the probability of the prophet to be wrong
        """
        ############### YOUR CODE GOES HERE ###############
        self.err_prob = err_prob

    def predict(self, y):
        """
        Predicts the label of the input point
        draws a random number between 0 and 1
        if the number is less than the probability, the prediction is correct (according to y)
        else the prediction is wrong
        NOTE: Realistically, the prophet should be a function from x to y (without getting y as an input)
        However, for the simplicity of our simulation, we will give the prophet y straight away
        :param y: the true label of the input point
        :return: a prediction for the label of the input point
        """
        ############### YOUR CODE GOES HERE ###############
        random_num = np.random.uniform(0,1,1)
        if random_num < self.err_prob:
            return y
        else:
            return abs(1-y)