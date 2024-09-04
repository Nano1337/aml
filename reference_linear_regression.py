import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial
from typing import NamedTuple
from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

"""
Overall observations: 
- It seems like functional programming way of programming is true to what Racket represented
"""

# lambda factory pattern for different losses
LOSS_FN_MAPPING = {
    "mae": lambda y_true, y_pred: jnp.mean(jnp.abs(y_true - y_pred)),
    "mse": lambda y_true, y_pred: jnp.mean((y_true - y_pred) ** 2),
    "rmse": lambda y_true, y_pred: jnp.sqrt(jnp.mean((y_true - y_pred) ** 2))
}

# define typing
class LinearParameters(NamedTuple):
    w: jnp.ndarray 
    b: jnp.ndarray | None

def linear_model(params: LinearParameters, x: jnp.ndarray) -> jnp.ndarray:
    # functional version of if-else to define whether linear model should include bias term
    return jax.lax.cond(
        jnp.isnan(params.b),
        lambda: jnp.dot(x, params.w),
        lambda: jnp.dot(x, params.w) + params.b
    )

# matrix-vector product
batched_linear_model = jax.vmap(linear_model, in_axes=(None, 0))

def loss_fn(loss_fn_arg, params: LinearParameters, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    # apply batched inference
    preds = batched_linear_model(params, x)

    # apply selected loss function
    return loss_fn_arg(y, preds)

@partial(jax.jit, static_argnames=('loss_fn_arg', 'learning_rate', ))
def update(learning_rate, loss_fn_arg, params, x, y):
    grad_loss_fn = jax.value_and_grad(partial(loss_fn, loss_fn_arg))
    loss, grad = grad_loss_fn(params, x, y)

    # TODO: don't know what this 
    return jax.tree.map(
        lambda p, g: p - g * learning_rate,
        params,
        grad
    ), loss

# Create Iterator pattern 
class BatchGenerator:
    def __init__(self, X, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.num_batches = (X.shape[0] - 1) // batch_size + 1

    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size

            yield self.X[start:end], self.Y[start:end]

class LinearRegression:
    def __init__(
            self,
            use_bias: bool = True
    ):
        self.use_bias = use_bias
        self.params = None

    def fit(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, **kwargs):
        loss = kwargs.get("loss", "mae")
        batch_size = kwargs.get("batch_size", 32)
        epochs = kwargs.get("epochs", 100)

        assert loss in LOSS_FN_MAPPING.keys(), f"loss must be one of {list(LOSS_FN_MAPPING.keys())}"

        number_of_features = x.shape[1]
        resolved_loss_fn = LOSS_FN_MAPPING[loss]

        batch_generator = BatchGenerator(x, y, batch_size)

        if self.use_bias:
            b = jnp.float32(1.0)
        else:
            b = jnp.nan

        w = jax.random.normal(jax.random.PRNGKey(42), (number_of_features,))

        self.params = LinearParameters(w, b)

        for epoch in tqdm(range(epochs), desc="Training"):
            for x_batch, y_batch in batch_generator:
                self.params, loss_value = update(
                    learning_rate,
                    resolved_loss_fn,
                    self.params,
                    x_batch,
                    y_batch
                )

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.params is not None, "Model not fitted yet"

        return batched_linear_model(self.params, jnp.asarray(x))

def main():
    xs, ys, coef = make_regression(
        n_features=1,
        n_informative=1,
        n_targets=1,
        n_samples=100_000,
        noise=2,
        coef=True,
        bias=5
    )

    scaler = StandardScaler()
    xs = scaler.fit_transform(xs)

    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=42)

    linear_regression = LinearRegression(use_bias=True)
    linear_regression.fit(x_train, y_train, loss="mae", learning_rate=0.5, epochs=1000, batch_size=512)

    y_predictions = linear_regression.predict(x_test)
    print(f"MAE: {mean_absolute_error(y_test, y_predictions)}")

    print(f"Linear Regression Params: {linear_regression.params}")

if __name__ == "__main__":
    main()
