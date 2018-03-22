import numpy as np
import pytest

@pytest.fixture(scope='module')
def data_preparation1():
    a1 = np.random.choice(np.arange(10), replace=True, size=[20]).astype(np.int32)
    a2 = np.random.choice(np.arange(10), replace=True, size=[20]).astype(np.int32)
    return a1, a2

@pytest.fixture(scope='class')
def data_preparation2():
    a1 = np.random.choice(np.arange(10), replace=True, size=[20]).astype(np.int32)
    a2 = np.random.choice(np.arange(10), replace=True, size=[20]).astype(np.int32)
    return a1, a2

@pytest.fixture(scope='function')
def data_preparation3():
    a1 = np.random.choice(np.arange(10), replace=True, size=[20]).astype(np.int32)
    a2 = np.random.choice(np.arange(10), replace=True, size=[20]).astype(np.int32)
    return a1, a2

class TestClass1:
    def test_case1(self, data_preparation1, data_preparation2, data_preparation3):
        a1, a2 = data_preparation1
        b1, b2 = data_preparation2
        c1, c2 = data_preparation3
        print('\nc1t1')
        print(a1)
        print(b1)
        print(c1)

    def test_case2(self, data_preparation1, data_preparation2, data_preparation3):
        a1, a2 = data_preparation1
        b1, b2 = data_preparation2
        c1, c2 = data_preparation3
        print('\nc1t2')
        print(a1)
        print(b1)
        print(c1)

class TestClass2:
    def test_case1(self, data_preparation1, data_preparation2, data_preparation3):
        a1, a2 = data_preparation1
        b1, b2 = data_preparation2
        c1, c2 = data_preparation3
        print('\nc2t1')
        print(a1)
        print(b1)
        print(c1)

    def test_case2(self, data_preparation1, data_preparation2, data_preparation3):
        a1, a2 = data_preparation1
        b1, b2 = data_preparation2
        c1, c2 = data_preparation3
        print('\nc2t2')
        print(a1)
        print(b1)
        print(c1)


