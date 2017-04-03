clear all; close all; clc;

% Adding Python module to Python search path
[own_path, ~, ~] = fileparts(mfilename('fullpath'));
module_path = fullfile(own_path, '..');
python_path = py.sys.path;
if count(python_path,module_path) == 0
    insert(python_path,int32(0),module_path);
end


a = py.array_test.array_return()
%x = py.numpy.random.random([4,4]);
data1 = double(py.array.array('d',py.numpy.nditer(a))); %Converting a numpy array to double array for matlab
%data = reshape(data,[4 4])';
%data2 = double(py.array.array('d',py.numpy.nditer(b)));
C = confusionmat(data1,data2)