import matplotlib.pyplot as plt

def plot_sample(data, base_filename):
    data = data.squeeze(1)  # remove the singleton dimension
    filenames = []
    # Iterate over the first dimension and plot each trace
    for i in range(data.shape[0]):  # Assuming 16 is the size of the first dimension
        plt.figure()
        for j in range(data.shape[2]):  # Assuming 200 is the size of the third dimension
            plt.plot(data[i, :, j].numpy(), label=f'Trace {j+1}')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title(f'Plot {i+1}')
        plt.legend()
        
        filename = f"{base_filename}_{i}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()  
    return filenames

def compare_samples(data_a, data_b, base_filename):
    data_a = data_a.squeeze(1)  # remove the singleton dimension
    data_b = data_b.squeeze(1)  # remove the singleton dimension
    filenames = []

    # Iterate over the first dimension and plot each pair of traces from data_a and data_b
    for i in range(data_a.shape[0]):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # Two subplots side by side
        # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))  # Two subplots, one on top of the other
        axes[0].set_title('Data A')
        axes[1].set_title('Data B')

        for j in range(data_a.shape[2]):
            axes[0].plot(data_a[i, :, j].numpy(), label=f'Trace {j+1}')
            axes[1].plot(data_b[i, :, j].numpy(), label=f'Trace {j+1}', linestyle='dashed')

        for ax in axes:
            ax.set_xlabel('t')
            ax.set_ylabel('y')
            ax.legend()

        filename = f"{base_filename}_{i}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig)
    return filenames
