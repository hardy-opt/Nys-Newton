data=importdata('rcv1_train_binary');
save('rcv1_train_binary', 'data');

fprintf('1st done!')
data=importdata('covtype_libsvm_binary_scale');
save('covtype_libsvm_binary_scale', 'data');
 
fprintf('2nd done!')
data=importdata('rcv1_test_binary');
save('rcv1_test_binary', 'data','-v7.3')
%save('rcv1_test_binary.mat', 'data', '-v7.3')

fprintf('3rd done!')