import numpy as np
import matplotlib.pyplot as plt
from dataset.fashion_mnist import load_fashion_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer


(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=False)

loss_per_epoch = []

network = DeepConvNet()

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20,
                  mini_batch_size=100,
                  optimizer='Adam',
                  optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)


trainer.train()

x = np.arange(len(trainer.train_acc_list))
plt.plot(x, trainer.train_acc_list, label='train acc')
plt.plot(x, trainer.test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("Fashion MNIST DeepConvNet Accuracy")
plt.show()


iter_per_epoch = int(trainer.iter_per_epoch)
for i in range(0, len(trainer.train_loss_list), iter_per_epoch):
    loss_per_epoch.append(
        np.mean(trainer.train_loss_list[i:i+iter_per_epoch])
    )

# Loss 그래프
x = np.arange(len(loss_per_epoch))
plt.plot(x, loss_per_epoch)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1.0)
plt.title("Training Loss per Epoch")
plt.grid()
plt.show()

print("--- Final Results ---")
print("Train Acc:", round(trainer.train_acc_list[-1] * 100, 2), "%  Test Acc:", round(trainer.test_acc_list[-1] * 100, 2), "%")