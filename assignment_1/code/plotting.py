import matplotlib.pyplot as plt
import csv

filename = '100_0.002_1500_200_200_mlp_numpy.csv'
step = []
loss = []
accuracy = []

with open(filename, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        step.append(int(row[0]))
        loss.append(float(row[1]))
        accuracy.append(float(row[2])*100)

plt.plot(step, loss, label='Loss')
plt.xlabel('step')
plt.ylabel('Loss')
plt.title('Loss changes in time')
plt.legend()
plt.show()