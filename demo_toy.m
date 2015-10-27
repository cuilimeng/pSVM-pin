clear all;
init();

%% load dataset

% heart
heart = load('./dataset/heart.dat');
bagsize = 4;
split.train_label = heart(:,size(heart,2)); split.train_label(split.train_label==1)=-1; split.train_label(split.train_label==2)=1;
data = heart(:,1:size(heart,2)-1);

%% kernal
kernel_type = 'linear';
trK = kernel_f(data, kernel_type);
teK = kernel_f(data, kernel_type); % in this toy example, we use training data for test
% trK = rbf_kernel2(data, data);
% teK = rbf_kernel2(data, data);


%% kernel InvCal
tic
para.method = 'InvCal';
para.C = 1;
para.ep = 0;
result_invcal = test_all_method(data, split, trK, teK, para);
toc

% kernel alter-pSVM with anealing
tic
para.C = 1; % empirical loss weight
para.C_2 = 1; % proportion term weight
para.ep = 0;
para.method = 'alter-pSVM';
N_random = 20;
result = [];
obj = zeros(N_random,1);
% stream=RandStream('mrg32k3a','Seed',2);%% generate stream for reproducibility of model
for pp = 1:N_random
%     set(stream,'Substream',pp);% the accuracy is strongly related to initlization
%     RandStream.setGlobalStream(stream);
    para.init_y = ones(length(trK),1);
    r = randperm(length(trK));
    para.init_y(r(1:floor(length(trK)/2))) = -1;
    result{pp} = test_all_method(data, split, trK, teK, para);
    obj(pp) = result{pp}.model.obj;
end
[mm,id] = min(obj);
result_alter = result{id};
toc

%% alter-pSVM with pinball loss
tic
para.C = 1; % empirical loss weight
para.C_2 = 1; % proportion term weight
para.ep = 0;
para.method = 'alter-pSVM-pin';
para.tau = 1;
N_random = 20;
result = [];
obj = zeros(N_random,1);
% stream=RandStream('mrg32k3a','Seed',2);%% generate stream for reproducibility of model
for pp = 1:N_random
%     set(stream,'Substream',pp);% the accuracy is strongly related to initlization
%     RandStream.setGlobalStream(stream);
    para.init_y = ones(length(trK),1);
    r = randperm(length(trK));
    para.init_y(r(1:floor(length(trK)/2))) = -1;
    result{pp} = test_all_method(data, split, trK, teK, para);
    obj(pp) = result{pp}.model.obj;
end
[mm,id] = min(obj);
result_alter_pin = result{id};
toc

%% regular SVM
tic
para.method = 'regularSVM';
result_regular = test_all_method(data, split, trK, teK, para);
toc

result_invcal
result_alter
result_alter_pin
result_regular

fprintf('%.2f(%.2f)\n',roundn(result_regular.train_acc,-2),roundn(result_regular.train_bag_error,-2));
fprintf('%.2f(%.2f)\n',roundn(result_invcal.train_acc,-2),roundn(result_invcal.train_bag_error,-2));
fprintf('%.2f(%.2f)\n',roundn(result_alter.train_acc,-2),roundn(result_alter.train_bag_error,-2));
fprintf('%.2f(%.2f)\n',roundn(result_alter_pin.train_acc,-2),roundn(result_alter_pin.train_bag_error,-2));
