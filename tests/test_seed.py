import torch
import pytorch_seed as rand
import random


def test_seed():
    N = 100
    rand.seed(5)
    r1 = torch.randn(N)
    rand.seed(5)
    r2 = torch.randn(N)
    assert torch.allclose(r1, r2)


def test_save_rng_unseeded_does_not_change_global():
    N = 100
    rand.seed(5)
    r1 = torch.randn(N)

    rand.seed(5)
    with rand.SavedRNG():
        torch.randn(N)
    r2 = torch.randn(N)
    assert torch.allclose(r1, r2)


def test_save_rng_seeded_does_not_change_global():
    N = 100
    rand.seed(5)
    r1 = torch.randn(N)

    rand.seed(5)
    with rand.SavedRNG(0):
        torch.randn(N)
    r2 = torch.randn(N)
    assert torch.allclose(r1, r2)


def test_save_rng_indendent_streams():
    rng_1 = rand.SavedRNG(0)
    rng_2 = rand.SavedRNG(1)

    with rng_2:
        rng_2_together_1 = torch.randn(1)
        rng_2_together_2 = torch.randn(1)

    with rng_1:
        rng_1_together_1 = torch.randn(1)
        rng_1_together_2 = torch.randn(1)

    rng_1 = rand.SavedRNG(0)
    rng_2 = rand.SavedRNG(1)

    # should be same as
    with rng_2:
        rng_2_interleaved_1 = torch.randn(1)

    with rng_1:
        rng_1_interleaved_1 = torch.randn(1)

    with rng_2:
        rng_2_interleaved_2 = torch.randn(1)

    with rng_1:
        rng_1_interleaved_2 = torch.randn(1)

    assert rng_2_together_1 == rng_2_interleaved_1
    assert rng_2_together_2 == rng_2_interleaved_2
    assert rng_1_together_1 == rng_1_interleaved_1
    assert rng_1_together_2 == rng_1_interleaved_2


def test_python_rng_is_saved():
    rand.seed(1)
    # restores random's RNG state
    with rand.SavedRNG():
        a = random.random()
    b = random.random()
    assert a == b

    rng = rand.SavedRNG(5)
    a = []
    with rng:
        a.append(random.random())
    random.random()
    with rng:
        a.append(random.random())
    rand.seed(5)
    b = [random.random() for _ in range(2)]
    assert a == b


if __name__ == "__main__":
    test_seed()
    test_save_rng_seeded_does_not_change_global()
    test_save_rng_unseeded_does_not_change_global()
    test_save_rng_indendent_streams()
    test_python_rng_is_saved()
