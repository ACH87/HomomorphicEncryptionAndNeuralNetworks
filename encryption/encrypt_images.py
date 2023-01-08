import Security_Functions as sf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)

pal = sf.PallierFunctions(min=50, max=100, keys={'public_key': {'n': 713, 'g': 16}, 'private_key':  {'lambda': 330, 'mu': 139}})
print(x_train[0])

x = pal.encrypt_numpy(x_train[0])

# enc_images_train= []
# enc_images_test = []
counter = 3241
for x in x_train[:5000]:
    enc_images_train = pal.encrypt_numpy(x)
    array_img = array_to_img(enc_images_train)

    plt.imshow(array_img)
    plt.savefig('data/train_2/' + str(counter) + '.jpeg')
    counter+=1
