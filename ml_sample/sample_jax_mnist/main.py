import time
import itertools

import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import optimizers, stax

import datasets as datasets


init_random_params, predict = stax.serial(
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(10),
    stax.LogSoftmax
)


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds*targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


def main():
    rng = random.PRNGKey(0)

    batch_size = 128
    step_size = 0.001
    num_epochs = 10
    momentum_mass = 0.9

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    # define data stream
    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_indices = perm[i*batch_size:(i+1)*batch_size]
                yield train_images[batch_size], train_labels[batch_indices]
    batches = data_stream()

    # define optimizer
    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    _, init_params = init_random_params(rng, (-1, 28*28))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print('\nStarting training...')
    for epoch in range(num_epochs):
        start_tm = time.time()
        for _ in range(num_epochs):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_tm = time.time() - start_tm
        
        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print(f'Epoch {epoch} in {epoch_tm:0.2f} sec')
        print(f'Training set accuracy {train_acc}')
        print(f'Test set accuracy {test_acc}')
    print('DONE')


if __name__ == '__main__':
    main()
