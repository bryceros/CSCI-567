import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    '''raise Exception(
             'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')'''
    N,D = x.shape
    centers = np.array(get_lloyd_k_means(len(x), 1, x, generator))
    while centers.shape[0] < n_cluster:
        centroids = x[centers, ]
        mask = np.ones(N,dtype=bool)
        mask[centers] = False
        temp = np.broadcast_to(centroids,shape=[N-centers.shape[0],centroids.shape[0],centroids.shape[1]])
        temp2 = np.linalg.norm(x[mask,None]-temp,axis=2)
        y = np.argmin(temp2,axis=1)
        r = np.zeros((y.size, centers.shape[0]))
        r[np.arange(y.size),y] = 1

        dis_x = np.linalg.norm(r-temp2,axis=1)
        dis_x = dis_x/ np.sum(dis_x)
        centers = np.append(centers,np.argmax(dis_x))

    
    centers = centers.tolist()
    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator
    def fit(self, x, centroid_func=get_lloyd_k_means, centroids = None):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        '''raise Exception(
             'Implement fit function in KMeans class')'''
        if centroids is None:             
            centroids = x[self.centers, ]
        j = np.Infinity
        for _ in range(self.max_iter):
            temp = np.broadcast_to(centroids,shape=[N,centroids.shape[0],centroids.shape[1]])
            temp2 = np.linalg.norm(x[:,None]-temp,axis=2)
            y = np.argmin(temp2,axis=1)
            r = np.zeros((y.size, self.n_cluster))
            r[np.arange(y.size),y] = 1

            j_new = np.mean(r*temp2,axis=None)
            if np.abs(j-j_new) <= self.e:
                break
            else:
                j = j_new
                centroids = np.matmul(r.T,x)/(np.sum(r,axis=0)[:,None])



        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        '''raise Exception(
             'Implement fit function in KMeansClassifier class')'''
        kmean = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e, generator=self.generator)
        centroids, min_centroid, _ = kmean.fit(x=x,centroid_func=centroid_func)
        centroid_labels = np.zeros([self.n_cluster])
        for index in range(self.n_cluster):
            y_sub = y[np.where(min_centroid == index)] 
            if y_sub != np.array([]):
                counts = np.bincount(y_sub)
                centroid_labels[index] = np.argmax(counts)

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        '''raise Exception(
             'Implement predict function in KMeansClassifier class')'''
        '''temp = np.broadcast_to(self.centroids,shape=[N,self.centroids.shape[0],self.centroids.shape[1]])
        temp2 = np.linalg.norm(x[:,None]-temp,axis=2)
        min_centroid = np.argmin(temp2,axis=1)'''
        kmean = KMeans(n_cluster=self.n_cluster, max_iter=1, e=0.0, generator=self.generator)
        _, min_centroid, _ = kmean.fit(x=x,centroid_func=get_lloyd_k_means,centroids=self.centroids)
        labels = self.centroid_labels[min_centroid]
        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    '''raise Exception(
             'Implement transform_image function')'''
    W,H,D = image.shape
    temp = np.broadcast_to(code_vectors,shape=[W,H,code_vectors.shape[0],code_vectors.shape[1]])
    temp2 = np.linalg.norm(image[:,:,None]-temp,axis=3)
    y = np.argmin(temp2,axis=2)
    new_im = code_vectors[y]
    

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

