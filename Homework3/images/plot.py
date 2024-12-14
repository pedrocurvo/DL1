import matplotlib.pyplot as plt
import pandas as pd

def plot(file):
    # Load the csv
    data = pd.read_csv(file)

    # Plot the data
    plt.figure(figsize=(10, 5))

    plt.plot(data['Step'].to_numpy(), data['Value'].to_numpy())
    plt.grid()
    plt.xlabel('Step', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('Bits per dimension', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(file.replace('.csv', '.pdf'))
    plt.show()

if __name__ == '__main__':
    plot('train_bpd.csv')
