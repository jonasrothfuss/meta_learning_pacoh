import ray
ray.init(redis_address="shallow:6379")
print('connected to ray head')

import time
import numpy as np

mu = 0
sig = 1

func = lambda x: np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

iter = 5

def f1(mu):
    means = []
    for j in range(iter):
        a = np.random.normal(loc=mu, size=5000000)
    means.append(a)
    return np.mean(means)

@ray.remote
def f2(mu):
    means = []
    for j in range(iter):
        a = np.random.normal(loc=mu, size=5000000)
    means.append(a)
    return np.mean(means)


@ray.remote
def f():
    time.sleep(0.01)
    return ray.services.get_node_ip_address()

print(set(ray.get([f.remote() for _ in range(100)])))

# t = time.time()
# # The following takes ten seconds.
# print([f1(i) for i in range(200)])


t = time.time()

# The following takes one second (assuming the system has at least ten CPUs).
print(ray.get([f2.remote(i) for i in range(2000)]))

print(time.time() - t)