import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

im_size = 100
channel_size = 3  # RGB format

#  number of iterations during data augmentation
augment_it = 10
split_ratios = (0.9, 0.95, 1)

#  number of images in each class
durer_count = 328*augment_it
vg_count = 877*augment_it
picasso_count = 439*augment_it
degas_count = 702*augment_it

durer_train = np.load('../data/conv_image_npy/durer.npy')[0:int(durer_count*split_ratios[0]) - 1, :, :, :]
durer_validation = np.load('../data/conv_image_npy/durer.npy')\
    [int(durer_count*split_ratios[0]): int(durer_count*split_ratios[1]) - 1, :, :, :]

vg_train = np.load('../data/conv_image_npy/vg.npy')[0:int(vg_count*split_ratios[0]) - 1, :, :, :]
vg_validation = np.load('../data/conv_image_npy/vg.npy')\
    [int(vg_count*split_ratios[0]): int(vg_count*split_ratios[1]) - 1, :, :, :]

picasso_train = np.load('../data/conv_image_npy/picasso.npy')[0:int(picasso_count*split_ratios[0]) - 1, :, :, :]
picasso_validation = np.load('../data/conv_image_npy/picasso.npy')\
    [int(picasso_count*split_ratios[0]): int(picasso_count*split_ratios[1]) - 1, :, :, :]

degas_train = np.load('../data/conv_image_npy/degas.npy')[0:int(degas_count*split_ratios[0]) - 1, :, :, :]
degas_validation = np.load('../data/conv_image_npy/degas.npy')\
    [int(degas_count*split_ratios[0]): int(degas_count*split_ratios[1]) - 1, :, :, :]

training_data = np.concatenate((durer_train[:, :, :, :], vg_train[0:2951, :, :, :], picasso_train[0:2951, :, :, :], degas_train[0:2951, :, :, :]), axis=0)/255

#print(durer_train.shape)
#print(durer_validation.shape)
#print(vg_train.shape)
#print(vg_validation.shape)
#print(picasso_train.shape)
#print(picasso_validation.shape)
#print(degas_train.shape)
#print(degas_validation.shape)

########################################################################################################################

layer_count_1 = 12
filter_width_1 = 5
stride_1 = 5
im_size_1 = int((im_size - filter_width_1)/stride_1) + 1

layer_count_2 = 48
filter_width_2 = 2
stride_2 = 2
im_size_2 = int((im_size_1 - filter_width_2)/stride_2) + 1

fully_connected_count = im_size_2*im_size_2*layer_count_2
output_count = 4

W1 = np.random.randn(filter_width_1, filter_width_1, channel_size, layer_count_1)*1e-2
W2 = np.random.randn(filter_width_2, filter_width_2, layer_count_1, layer_count_2)*1e-2
W3 = np.random.randn(fully_connected_count, output_count)*1e-2

b1 = np.random.randn(layer_count_1)*1e-2
b2 = np.random.randn(layer_count_2)*1e-2
b3 = np.random.randn(output_count)*1e-2

model = [
    {'layer_type': 'conv', 'W': W1, 'b': b1, 'activation': 'relu', 'filter_width': filter_width_1, 'stride': stride_1, 'layer_count': layer_count_1, 'im_size': im_size_1},
    {'layer_type': 'conv', 'W': W2, 'b': b2, 'activation': 'relu', 'filter_width': filter_width_2, 'stride': stride_2, 'layer_count': layer_count_2, 'im_size': im_size_2},
    {'layer_type': 'flatten'},
    {'layer_type': 'dense', 'W': W3, 'b': b3, 'activation': 'softmax'}
]


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))


def relu_bp(z, x):
    dz = np.array(x)
    dz[z <= 0] = 0
    return dz


def softmax_bp(label, prediction):
    return prediction - label


def softmax_loss(label, prediction):
    return -np.log(np.sum(softmax(prediction)*label))


def conv3d(X, W, b, lc, stride, im_size):
    z = np.zeros((im_size, im_size, lc))
    for i in range(lc):
        for j in range(X.shape[2]):
            z[:, :, i] = z[:, :, i] + convolve2d(X[:, :, j], W[:, :, j, i], mode='valid')[::stride, ::stride]
        z[:, :, i] = z[:, :, i] + b[i]
    return z


def forward_prop(X, model):
    L = len(model)
    forward_memory = {}
    X_curr = X
    z_curr = X
    forward_memory['X0'] = X_curr

    for i in range (0, L):
        if model[i]['layer_type'] == 'conv':
            W = model[i]['W']
            b = model[i]['b']
            fw = model[i]['filter_width']
            stride = model[i]['stride']
            lc = model[i]['layer_count']
            im_size = model[i]['im_size']
            activation = model[i]['activation']

            z = conv3d(X_curr, W, b, lc, stride, im_size)
            X_next = relu(z)

            forward_memory['X' + str(i+1)] = X_next
            forward_memory['Z' + str(i+1)] = z

            z_curr = z
            X_curr = X_next

        elif model[i]['layer_type'] == 'flatten':
            X_curr = X_curr.flatten()
            z_curr = z_curr.flatten()
            forward_memory['X' + str(i+1)] = X_curr
            forward_memory['Z' + str(i+1)] = z_curr

        else:
            W = model[i]['W']
            b = model[i]['b']
            z = np.matmul(W.T, X_curr) + b
            X_next = softmax(z)
            forward_memory['X' + str(i+1)] = X_next
            forward_memory['Z' + str(i+1)] = z

            X_curr = X_next

    return X_curr, forward_memory


def back_prop(output_layer, forward_memory, label, model):
    L = len(model)
    gradients = {}

    dz = softmax_bp(label, output_layer)
    da = 1

    for i in reversed(range(0, L)):
        if model[i]['layer_type'] == 'dense':
            W = model[i]['W']
            b = model[i]['b']
            da = np.matmul(W, dz)

            a_prev = forward_memory['X' + str(i)]
            gradients['dW' + str(i)] = np.matmul(a_prev.reshape(-1, 1), dz.reshape(1, -1))
            gradients['db' + str(i)] = dz

        elif model[i]['layer_type'] == 'flatten':
            da = da.reshape(model[i-1]['im_size'], model[i-1]['im_size'], model[i-1]['layer_count'])

        else:
            W = model[i]['W']
            b = model[i]['b']
            fw = model[i]['filter_width']
            stride = model[i]['stride']
            lc = model[i]['layer_count']
            im_size = model[i]['im_size']

            z_curr = forward_memory['Z' + str(i+1)]
            dz = relu_bp(z_curr, da)
            a_prev = forward_memory['X' + str(i)]
            grad_vect_w = np.zeros(W.shape)
            grad_vect_b = np.zeros(b.shape)
            if i > 0:
                da = np.zeros((model[i - 1]['im_size'], model[i - 1]['im_size'], model[i - 1]['layer_count']))
            for j in range(lc):
                for l in range(im_size):
                    for k in range(im_size):
                        grad_vect_w[:, :, :, j] = grad_vect_w[:, :, :, j] + dz[l, k, j]*a_prev[stride*l:stride*(l+1), stride*k:stride*(k+1), :]
                        grad_vect_b[j] = grad_vect_b[j] + dz[l, k, j]
                        if i > 0:
                            da[stride*l:stride*(l+1), stride*k:stride*(k+1), :] = da[stride*l:stride*(l+1), stride*k:stride*(k+1), :] + W[:, :, :, j]*dz[l, k, j]

            gradients['dW' + str(i+1)] = grad_vect_w
            gradients['db' + str(i+1)] = grad_vect_b

    return gradients


def iterate(lr, rc, model, gradients):
    L = len(model)
    for i in range(0, L):
        if i < (L-2):
            model[i]['W'] = (1 - lr*rc)*model[i]['W'] - lr*gradients['dW' + str(i+1)]
            model[i]['b'] = (1 - lr*rc)*model[i]['b'] - lr*gradients['db' + str(i+1)]
        elif i == (L-2):
            continue
        else:
            model[i]['W'] = (1 - lr*rc)*model[i]['W'] - lr*gradients['dW' + str(i)]
            model[i]['b'] = (1 - lr*rc)*model[i]['b'] - lr*gradients['db' + str(i)]


def train(training_data, model, lr, rc, num_epoch):
    tr_err = []
    temp_loss = 0
    temp_out_layer = np.zeros(4)
    plot_freq = int(num_epoch/50)
    for i in range(num_epoch):
        index = np.random.randint(training_data.shape[0])
        durer_flag = int(index < 2951)
        vg_flag = int(index >= 2951 and index < 2951 + 2950)
        picasso_flag = int(index >= 2951 + 2950 and index < 2951 + 2950 + 2950)
        degas_flag = int(index >= 2951 + 2950 + 2950)
        label = durer_flag*np.array([1, 0, 0, 0]) + vg_flag*np.array([0, 1, 0, 0]) + picasso_flag*np.array([0, 0, 1, 0]) + degas_flag*np.array([0, 0, 0, 1])
        x = training_data[index, :, :, :]

        output_layer, forward_memory = forward_prop(x, model)
        gradients = back_prop(output_layer, forward_memory, label, model)
        iterate(lr, rc, model, gradients)
        temp_loss = temp_loss + softmax_loss(label, output_layer)
        temp_out_layer = temp_out_layer + output_layer
        if i % plot_freq == 0:
            print('%', 100*i/num_epoch, ' Label: ', label, ' Softmax: ', temp_out_layer/plot_freq)
            tr_err.append(temp_loss/plot_freq)
            temp_loss = 0
            temp_out_layer = np.zeros(4)

    plt.plot(tr_err)
    plt.show()


lr = 1e-4
rc = 0
num_epoch = 50000
train(training_data, model, lr, rc, num_epoch)








