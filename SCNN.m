% last modified on Jun,2nd, 2017 by Lin
clear

% load maize_1y
data = xlsread('E:\����-Chestnut\PCA��������-528.xlsx');
x_train = data(1:352,1:20);
x_test = data(353:end,1:20);
y_train = data(1:352,21);
y_test = data(353:end,21);

ntr = size(x_train,1);
nte = size(x_test,1);

x_train = x_train';   %ת��֮���б�ʾ�������б�ʾ�۲�����
x_test = x_test';     %ת��֮���б�ʾ�������б�ʾ�۲�����
[inputn, inputps] = mapminmax(x_train);   %inputn�Ǿ�����һ��������ݣ�inputps�ǹ�һ�������еĲ�����ÿ�������ľ�ֵ����׼�
inputn_test = mapminmax('apply', x_test, inputps);  %��Ԥ�⼯����ѵ������������Ĳ������й�һ������

Q = minmax(inputn);
net = newc(Q,2);
net = init(net);
net.trainparam.epochs = 100; % ѵ������
net = train(net,inputn);

a = sim(net,inputn_test);
ac = vec2ind(a);
ncor = 0;    % number of correct predictions
for i=1:nte
    if isequal(ac(i),y_test(i))
        ncor = ncor + 1;
    end
end
accuracy_pred = ncor / nte * 100;
fprintf(1,'Ԥ���׼ȷ���ǣ� %4.2f%% \n', accuracy_pred);

Y = sim(net,inputn);
yc = vec2ind(Y);
ncor = 0;    % number of correct predictions
for i=1:ntr
    if isequal(yc(i),y_train(i))
        ncor = ncor + 1;
    end
end
accuracy_return = ncor / ntr * 100;
fprintf(1,'���е�׼ȷ���ǣ� %4.2f%% \n', accuracy_return);

% d = zeros(8,8);
% for i=1:ntr
%     d(y_train(i),yc(i)) = d(y_train(i),yc(i)) + 1;
% end
