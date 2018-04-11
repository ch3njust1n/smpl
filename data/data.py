'''
	Justin Chen

	6.19.17
'''
import matplotlib.pyplot as plt
plt.switch_backend('agg')


'''
Visualize a batch of MNIST digits

Input: samples (torch.FloatTensor) Tensor of batch of MNIST images
       targets (torch.FloatTensor) Tensor of batch of MNIST labels
       dimn    (tuple) Tuple representing image dimensions (optional)
'''
def visualize(samples, labels, dim=(28,28)):
    # first index is sample in batch
    batch_size = len(samples)
    for s in range(batch_size):
        pixels = samples[s][0].numpy().reshape(dim)
        plt.title('Label: {label}'.format(label=label[s]))
        plt.imshow(pixels, cmap='gray')
        plt.show()


'''
Visualize data in 2D

Inputs: x_data  (list)
        y_data  (list) (optional)
        title   (string) (optional)
        x_label (string) (optional)
        y_label (string) (optional)
'''
def visualize_2D(x_data, y_data=[], title='', x_label='', y_label=''):
    fig = plt.figure()
    plot = fig.add_subplot(111)

    if len(y_data) == 0:
        plot.plot(x_data)
    else:
        plot.plot(x_data, y_data)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


'''
Visualize multiple plots in 2D

Input: numplots      (int)
       title         (string)
       x_data_list   (list) (optional)
       y_data_list   (list) (optional)
       subplt_titles (list) (optional)
       x_label       (list) (optional)
       y_label       (list) (optional)
'''
def subplots_2D(num_plots, title, x_data_list=[], y_data_list=[], subplt_titles=[], x_label=[], y_label=[]):
    f, axarr = plt.subplots(num_plots, sharex=False)
    subplt_titles[0] = ' '.join([title, subplt_titles[0]])

    if len(x_data_list) == 0:
        for i, p in enumerate(axarr):
            p.set_title(subplt_titles[i])
            p.set_xlabel(x_label[i])
            p.set_ylabel(y_label[i])
            p.plot(y_data_list[i])
    else:
        for i, p in enumerate(axarr):
            p.set_title(subplt_titles[i])
            p.set_xlabel(x_label[i])
            p.set_ylabel(y_label[i])
            p.plot(x_data_list[i], y_data_list[i])

    plt.tight_layout()
    plt.show()