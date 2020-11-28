import random
import numpy as np

class BloomFilter:
    def __init__(self, filter_size, hash_size, debug=False):
        self.debug = debug
        self.filter_size = filter_size
        self.hash_size = hash_size
        self.bitarray = [0] * filter_size
        self.hashes = [self.randomHash() for _ in range(hash_size)]

    def randomHash(self):
        """
        Initialize random Hash functions
        """
        modulus = self.filter_size
        a, b = random.randint(1, modulus - 1), random.randint(1, modulus - 1)

        def f(x):
            return hash(x) % (a + b) % modulus

        return f

    def train_bloom_filter(self, train_data):
        """
        Train Bloom Filter for given data (users-bots)
        """
        for val in train_data:
            if self.debug:
                print('val: ', val)
            for i in range(0, self.hash_size):
                k = self.hashes[i](val[0])
                if self.debug:
                    print('k: ', k)
                self.bitarray[k] = 1
            if self.debug:
                print('___end training____')

    def is_in_bloom_filter(self, value):
        """
    check if given user is a bot
    by checking if this user exists in the trained Bloom Filter
    """
        for i in range(0, self.hash_size):
            k = self.hashes[i](value)
            if self.debug:
                print(k)
            if self.bitarray[k] == 0:
                unique, counts = np.unique(self.bitarray, return_counts=True)
                if self.debug:  # print how many 0 and 1 are in bitarray
                    print(dict(zip(unique, counts)))
                return False
        if self.debug:  # print how many 0 and 1 are in bitarray
            unique, counts = np.unique(self.bitarray, return_counts=True)
            print(dict(zip(unique, counts)))
        return True
