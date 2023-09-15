from _perceptron.perceptron import Perceptron

def _activator(x):
    return 1 if x > 0 else 0

def get_and_training_dataset():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perceptron():
    p = Perceptron(2 , _activator)
    input_vecs, labels = get_and_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == "__main__":
    and_perceptron = train_and_perceptron()
    print(and_perceptron)
    print("测试：\n")
    print(and_perceptron.predict([1, 1]))