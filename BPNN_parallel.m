% 使用多个CPU核心并行进行神经网络建模与仿真计算
% last modified on Jun, 27th, 2017 by Lin
clear

% load data
% data = xlsread('.\板栗.xlsx','特征波长数据');
% x_train = data(1:352,1:20);
% x_test = data(353:end,1:20);
% y_train = data(1:352,21);
% y_test = data(353:end,21);
data = xlsread('G:\板栗\板栗-文章\特征波长-528.xlsx');
x_train = data(1:352,1:20);
x_test = data(353:end,1:20);
y_train = data(1:352,21);
y_test = data(353:end,21);

nhl = 2; % 隐藏层层数

ntr = size(x_train,1);
nte = size(x_test,1);
nx = size(x_train,2);

x_train = x_train';   %转置之后行表示变量，列表示观测样本
x_test = x_test';     %转置之后行表示变量，列表示观测样本
y_train = y_train';    %转置之后行表示输出（响应），列表示观测样本
y_test = y_test';      %转置之后行表示输出（响应），列表示观测样本
[inputn, inputps] = mapminmax(x_train);   %inputn是经过归一化后的数据，inputps是归一化过程中的参数（每个变量的均值及标准差）
[outputn, outputps] = mapminmax(y_train);
hiddenLayer = ones(1,nhl);
hiddenLayer = hiddenLayer * 1;   % hiddenLayer表示隐藏层的层数及每层神经元的数量
net = feedforwardnet(hiddenLayer,'trainlm');  % 也可以用net = newff(inputn, outputn, hiddenLayer); 
[net, tr] = train(net, inputn, outputn, 'Useparallel','yes');

inputn_test = mapminmax('apply', x_test, inputps);  %将预测集按照训练集输入变量的参数进行归一化处理
an = sim(net, inputn_test, 'Useparallel','yes');   
BPoutput = mapminmax('reverse', an, outputps);    %将预测结果按照训练集输出响应的参数进行逆归一化处理
BPoutput = round(BPoutput);
ncor = 0;    % number of correct predictions
for i=1:nte
    if isequal(BPoutput(:,i),y_test(:,i))
        ncor = ncor + 1;
    end
end
accuracy_pred = ncor / nte * 100;
fprintf(1,'预测的准确率是： %4.2f%% \n', accuracy_pred);

an = sim(net, inputn, 'Useparallel','yes');   
BPoutput = mapminmax('reverse', an, outputps);    %将预测结果按照训练集输出响应的参数进行逆归一化处理
BPoutput = round(BPoutput);
ncor = 0;    % number of correct predictions
for i=1:ntr
    if isequal(BPoutput(:,i),y_train(:,i))
        ncor = ncor + 1;
    end
end
accuracy_return = ncor / ntr * 100;
fprintf(1,'回判的准确率是： %4.2f%% \n', accuracy_return);
% mesh(x,y,z)