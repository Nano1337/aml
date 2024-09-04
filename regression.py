import jax
from utils import *
from jax.random import randint, normal as randn
import jax.numpy as np 
from jax.scipy.linalg import eigh, solve

'''
Implement the sigmoid function. Be sure to do this in a vectorized way, since `a` is an arbitrary array.
'''
@jax.jit
@jax.vmap
def sigmoid(a):
    return 1 / (1 + np.exp(-a))
#

'''
A class for logistic regression
'''
class LogisticRegression:
    '''
    The X_train and Y_train are instance attributes containing training data for, respectively, input features (flattened images) and corresponding binary (0, 1) labels.
    Similarly for X_val and Y_val, for validation data (used for testing generalization).
    `beta` is the scale applied to the L2 regularization
    '''
    def __init__(self, X_train, Y_train, X_val, Y_val, beta = 1e-1):
        self.X_train = self.augment_for_bias(X_train)
        self.Y_train = Y_train

        self.N = self.X_train.shape[0]
        self.D = self.X_train.shape[1]

        self.X_val = self.augment_for_bias(X_val)
        self.Y_val = Y_val

        self.beta = beta

        self.key = jax.random.PRNGKey(0)
        #

    '''
    Return an initialization of the weight vector.
    This should be initialized such that the following holds: taking the dot product between (1) the initialized weight vector and (2) an input vector whose entries are i.i.d. samples from a standard normal distribution gives, in expectation, a value of 1.
    '''
    def initialization(self):
        w = randn(self.key, (self.D,))
        return w / np.linalg.norm(w)
    #

    '''
    This will take care of the bias term for you.
    '''
    def augment_for_bias(self, X):
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
    #

    '''
    Compute and return the Lipschitz constant for logistic regression (see class slides).
    '''
    def lipschitz_constant(self):
        # 1/4 * eigenvalue_max of X_train^T @ X_train
        return 0.25 * np.max(eigh(self.X_train.T @ self.X_train)[0])
    #

    '''
    Compute and return the loss (NLL) averaged over the training dataset, given weight vector `w`.
    '''
    def train_loss(self, w):
        # -1/N * sum(y_i * log(sigmoid(w^T x_i)) + (1 - y_i) * log(1 - sigmoid(w^T x_i)))
        return -1/self.N * np.sum(self.Y_train * np.log(sigmoid(self.X_train @ w)) + (1 - self.Y_train) * np.log(1 - sigmoid(self.X_train @ w)))
    #

    '''
    Compute and return the loss (NLL) averaged over the validation dataset, given weight vector `w`.
    '''
    def validation_loss(self, w):
        return -1/self.N * np.sum(self.Y_val * np.log(sigmoid(self.X_val @ w)) + (1 - self.Y_val) * np.log(1 - sigmoid(self.X_val @ w)))
    #

    '''
    Compute and return the prediction accuracy of the model with weight vector `w` on the validation data.
    Given the sigmoid output, you should output a label of 1 if the output is at least 0.5, and a label of 0 otherwise.
    '''
    def validation_accuracy(self, w):
        # 1/N * sum(y_i == 1 if sigmoid(w^T x_i) >= 0.5 else 0)
        return 1/self.N * np.sum(self.Y_val == (sigmoid(self.X_val @ w) >= 0.5))
    #

    '''
    Compute the loss, and the loss gradient, for the model with weight vector `w`.
    The loss & gradient should be restricted to `data_samples`. If set to None, then use all of the data. Otherwise, `data_samples` contains an array of integers, which should be used to index into X_train and Y_train to obtain input vectors and class labels.
    The `reduce` parameter is a boolean, indicating whether or not to average the gradients over data samples -> set to False for SAGA initialization
    The train loss, and gradient, should be returned as a 2-tuple.
    '''
    def train_loss_and_grad(self, w, data_samples=None, reduce=True):
        if data_samples is None:
            X = self.X_train
            Y = self.Y_train
            N = self.N
        else:
            # Ensure data_samples is an array of integers
            data_samples = np.array(data_samples, dtype=np.int32)
            X = self.X_train[data_samples]
            Y = self.Y_train[data_samples]
            N = len(data_samples)

        activations = sigmoid(X @ w)
        loss = -1/N * np.sum(Y * np.log(activations) + (1 - Y) * np.log(1 - activations))
        
        # Add L2 regularization to the loss
        loss += 0.5 * self.beta * np.sum(w**2)
        
        # Compute gradient
        gradient = jax.grad(lambda w: loss, argnums=0)(w)

        if not reduce:
            gradient *= N
        
        return loss, gradient
    #

    '''
    Compute the loss, and the direction derived by Newton's method, for the model with weight vector `w`.
    Note: this requires computing the Hessian for logistic regression, and solving a linear system.
    The expected inputs and outputs are the same as in `train_loss_and_grad` (except no reduce).
    '''
    def train_loss_and_newton(self, w, data_samples):
        # TODO: compute gradient
        gradient = 0
        hessian = jax.hessian(lambda w: self.train_loss(w), argnums=0)(w)

        newton_direction = solve(hession, -gradient)
    #
#

if __name__ == "__main__":

    def create_random_dataset(N, D, key):
        key1, key2 = jax.random.split(key)
        X = jax.random.normal(key1, (N, D))
        Y = jax.random.randint(key2, (N,), 0, 2)
        return X, Y

    # test sigmoid function
    X = np.arange(10)
    print(f"sigmoid test: \nInputs: {X} \nOutputs: {sigmoid(X)}\n")

    # create LogisticRegression class
    # create a random dataset
    key = jax.random.PRNGKey(1234)
    X_train, Y_train = create_random_dataset(10000, 10, key)
    X_val, Y_val = create_random_dataset(100, 10, key)
    model = LogisticRegression(X_train, Y_train, X_val, Y_val)

    # Test initialization
    print("Testing initialization...")
    model = LogisticRegression(X_train, Y_train, X_val, Y_val)
    w = model.initialization()
    print(f"w shape: {w.shape}")

    # Generate a large number of random input vectors
    num_test_vectors = 10000
    test_vectors = jax.random.normal(jax.random.PRNGKey(5678), (num_test_vectors, model.D - 1))  # -1 because X_train includes bias

    # Compute dot products
    dot_products = test_vectors @ w[:-1]  # Exclude the bias term from w

    # Calculate mean of dot products
    mean_dot_product = np.mean(dot_products)
    
    print(f"Mean dot product: {mean_dot_product}")
    print(f"Is initialization correct? {np.abs(mean_dot_product - 1) < 1e-2}")

    # Additional test: verify that w is unit length
    print(f"Is w unit length? {np.abs(np.linalg.norm(w) - 1) < 1e-6}")
    # test train_loss_and_grad
    num_samples = 1000  # You can adjust this number as needed
    key = jax.random.PRNGKey(0)  # Use a fixed seed for reproducibility
    random_indices = jax.random.choice(key, model.N, shape=(num_samples,), replace=False)
    data_samples = (model.X_train[random_indices], model.Y_train[random_indices])
    
    loss, gradient = model.train_loss_and_grad(w, data_samples)
    print(f"Loss: {loss}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Gradient (first few elements): {gradient[:10]}")