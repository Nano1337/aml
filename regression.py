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

        #

    '''
    Return an initialization of the weight vector.
    This should be initialized such that the following holds: taking the dot product between (1) the initialized weight vector and (2) an input vector whose entries are i.i.d. samples from a standard normal distribution gives, in expectation, a value of 1.
    '''
    def initialization(self):
        w = randn(prng_key, (self.D,))
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
        eighnum, _ = eigh(np.dot(self.X_train.T, self.X_train))
        return 1/4 * np.max(eighnum)
    #

    '''
    Compute and return the loss (NLL) averaged over the training dataset, given weight vector `w`.
    '''
    def train_loss(self, w, X=None, Y=None):
        if X is None or Y is None:
            X, Y = self.X_train, self.Y_train
        N = X.shape[0]
        loss = -1/N * np.sum(Y * np.log(sigmoid(np.dot(X, w))) + 
                             (1 - Y) * np.log(1 - sigmoid(np.dot(X, w))))
        l2_loss = self.beta * np.linalg.norm(w)**2
        return loss + l2_loss
        
    #

    '''
    Compute and return the loss (NLL) averaged over the validation dataset, given weight vector `w`.
    '''
    def validation_loss(self, w):
        N_val = self.X_val.shape[0]
        loss = -1/N_val * np.sum(self.Y_val * np.log(sigmoid(np.dot(self.X_val, w))) + 
                                 (1 - self.Y_val) * np.log(1 - sigmoid(np.dot(self.X_val, w))))
        l2_loss = self.beta * np.linalg.norm(w)**2
        return loss + l2_loss
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

        # define single sample loss function
        def single_sample_loss_fn(w, x, y):
            logit = np.dot(x, w)
            log_likelihood = jax.nn.log_sigmoid(logit) * y + jax.nn.log_sigmoid(-logit) * (1 - y)
            return -log_likelihood + 0.5 * self.beta * np.sum(w**2)

        # vectorize functions
        loss_fn = jax.vmap(single_sample_loss_fn, in_axes=(None, 0, 0))
        grad_fn = jax.vmap(jax.grad(single_sample_loss_fn, argnums=0), in_axes=(None, 0, 0))

        # compute
        losses = loss_fn(w, X, Y)
        grads = grad_fn(w, X, Y)

        if reduce:
            return np.mean(losses), np.mean(grads, axis=0)
        else:
            return losses, grads
    #

    '''
    Compute the loss, and the direction derived by Newton's method, for the model with weight vector `w`.
    Note: this requires computing the Hessian for logistic regression, and solving a linear system.
    The expected inputs and outputs are the same as in `train_loss_and_grad` (except no reduce).
    '''
    def train_loss_and_newton(self, w, data_samples=None):
    
        if data_samples is None: 
            data_samples = np.arange(self.N)
        else: 
            data_samples = np.array(data_samples, dtype=np.int32)

        X = self.X_train[data_samples]
        Y = self.Y_train[data_samples]

        # compute loss and gradient
        grad_loss_fn = jax.value_and_grad(self.train_loss)
        hess_fn = jax.hessian(self.train_loss)
        loss, grad = grad_loss_fn(w, X, Y)
        hess = hess_fn(w, X, Y)

        newton_dir = solve(hess, grad)
        return loss, newton_dir
    #
#

if __name__ == "__main__":

    """
    Personal Testing Code
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Create sample dataset for testing
    xs, ys = make_classification(
        n_features=10,
        n_classes=2,
        n_samples=100_000,
        n_clusters_per_class=1,
        n_informative=1,
        random_state=42,
        flip_y=0.1
    )
    scaler = StandardScaler()
    xs = scaler.fit_transform(xs)
    X_train, X_val, Y_train, Y_val = train_test_split(xs, ys, test_size=0.2, random_state=42)
    model = LogisticRegression(X_train, Y_train, X_val, Y_val)

    # Test initialization
    print("Testing initialization:")
    w = model.initialization()
    num_test_vectors = 10000
    test_vectors = jax.random.normal(jax.random.PRNGKey(5678), (num_test_vectors, model.D - 1))  # -1 because X_train includes bias
    dot_products = np.dot(test_vectors, w[:-1]) #  test_vectors @ w[:-1]  # Exclude the bias term from w
    mean_dot_product = np.mean(dot_products)
    variance_dot_product = np.var(dot_products)
    print(f"   Mean: {mean_dot_product}, Variance: {variance_dot_product}")
    # Additional test: verify that w is unit length
    print(f"   Is w unit length? {np.abs(np.linalg.norm(w) - 1) < 1e-6}")

    # Test calculating lipshitz constant
    print("Testing Lipshitz calculation: ")
    l_const = model.lipschitz_constant()
    print(f"    Lipshitz constant is {l_const}")

    # Test train loss and grad: 
    print("Testing train loss and grad: ")
    print(f"    With reduction:")
    loss, grad = model.train_loss_and_grad(w, reduce=True)
    print(f"    Loss shape is: {loss.shape}, Gradient shape is: {grad.shape}")
    print(f"    Without reduction:")
    loss, grad = model.train_loss_and_grad(w, reduce=False)
    print(f"    Loss shape is: {loss.shape}, Gradient shape is: {grad.shape}")
    print(f"    Without reduction and with batch indices")
    random_indices = [0,4,2,3,65,1234,64]
    loss, grad = model.train_loss_and_grad(w, data_samples=random_indices,reduce=False)
    print(f"    Loss shape is: {loss.shape}, Gradient shape is: {grad.shape}")
    

    # Test train loss and grad: 
    print("Testing Newton loss and grad: ")
    print(f"    With reduction:")
    loss, grad = model.train_loss_and_newton(w)
    print(f"    Loss shape is: {loss.shape}, Gradient shape is: {grad.shape}")
    print(f"    With batch indices")
    random_indices = [0,4,2,3,65,1234,64]
    loss, grad = model.train_loss_and_newton(w, data_samples=random_indices)
    print(f"    Loss shape is: {loss.shape}, Gradient shape is: {grad.shape}")
    
