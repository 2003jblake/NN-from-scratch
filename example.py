import preprocessor as p
import neural_network as nn

x_test, y_test, x_train, y_train = p.process_data('heart.dat')
network = nn.NeuralNetwork()
l1 = nn.Layer(13, 8, nn.leaky_relu)
l3 = nn.Layer(8, 8, nn.leaky_relu)
l2 = nn.Layer(8, 1, nn.sigmoid)

network.add_layer(l1)
#n.add_layer(l3)
network.add_layer(l2)

network.train(x_train, y_train)
print(network.predict(x_test, y_test))