import matplotlib.pyplot as plt
import csv

filename = '100_0.002_1500_200_200_SGD_mlp.csv'
step = []
loss = []
accuracy = []

with open(filename, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        step.append(int(row[0]))
        loss.append(float(row[1]))
        accuracy.append(float(row[2])*100)

plt.plot(step, accuracy, label='Accuracy')
plt.xlabel('step')
plt.ylabel('Accuracy')
plt.title('Accuracy changes in time')
plt.legend()
plt.show()