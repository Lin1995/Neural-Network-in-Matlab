% last modified on Jun,3rd, 2017 by Lin
clear

% load maize_1y
data = xlsread('E:\文章-Chestnut\PCA特征波长-528.xlsx');
x_train = data(1:352,1:20);
x_test = data(353:end,1:20);
y_train = data(1:352,21);
y_test = data(353:end,21);

ntr = size(x_train,1);
nte = size(x_test,1);

x_train = x_train';   %转置之后行表示变量，列表示观测样本
x_test = x_test';     %转置之后行表示变量，列表示观测样本
[inputn, inputps] = mapminmax(x_train);   %inputn是经过归一化后的数据，inputps是归一化过程中的参数（每个变量的均值及标准差）
inputn_test = mapminmax('apply', x_test, inputps); 

net = newsom(minmax(inputn),[2 1],'hextop','boxdist');
net.trainparam.epochs = 100; % 训练次数
net = train(net,inputn);
% net = newsom(minmax(x_train),[2 4],'hextop','boxdist');
% net.trainparam.epochs = 100; % 训练次数
% net = train(net,x_train);

a = sim(net,inputn_test);
% a = sim(net,x_test);
ac = vec2ind(a);
ncor = 0;    % number of correct predictions
for i=1:nte
    if isequal(ac(i),y_test(i))
        ncor = ncor + 1;
    end
end
accuracy_pred = ncor / nte * 100;
fprintf(1,'预测的准确率是： %4.2f%% \n', accuracy_pred);

Y = sim(net,inputn);
% Y = sim(net,x_train);
yc = vec2ind(Y);
ncor = 0;    % number of correct predictions
for i=1:ntr
    if isequal(yc(i),y_train(i))
        ncor = ncor + 1;
    end
end
accuracy_return = ncor / ntr * 100;
fprintf(1,'回判的准确率是： %4.2f%% \n', accuracy_return);

d = zeros(8,8);
for i=1:nte
    d(y_test(i),yc(i)) = d(y_test(i),yc(i)) + 1;
end