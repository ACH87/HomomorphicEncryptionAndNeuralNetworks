requirements:
docker
python libraries:
keras
tensorflow
numpy
sklearn

security functions:
all security functions are within "security functions" and are tested within
security tests

All keras layers are written in the keras_layers code
Tests for layers are in the functions_test file

all neural network models are within the Lenet folder

Fully homomorphic encryption code can be found in:
PySEAL-master/SEALPythonExamples/
Key files: 
function_unit_test -> tests the convolution, dense and pooling operations
main.py -> runs the run function
my_functions -> each operation

to run first run the build-docker.sh script
then the python script can be run using the run-docker.sh
to chage the file to run change the file listed in that file