% ʹ�ö��CPU���Ĳ��н��������罨ģ��������
% last modified on Jun, 27th, 2017 by Lin
clear

% load data
% data = xlsread('.\����.xlsx','������������');
% x_train = data(1:352,1:20);
% x_test = data(353:end,1:20);
% y_train = data(1:352,21);
% y_test = data(353:end,21);
data = xlsread('G:\����\����-����\��������-528.xlsx');
x_train = data(1:352,1:20);
x_test = data(353:end,1:20);
y_train = data(1:352,21);
y_test = data(353:end,21);

nhl = 2; % ���ز����

ntr = size(x_train,1);
nte = size(x_test,1);
nx = size(x_train,2);

x_train = x_train';   %ת��֮���б�ʾ�������б�ʾ�۲�����
x_test = x_test';     %ת��֮���б�ʾ�������б�ʾ�۲�����
y_train = y_train';    %ת��֮���б�ʾ�������Ӧ�����б�ʾ�۲�����
y_test = y_test';      %ת��֮���б�ʾ�������Ӧ�����б�ʾ�۲�����
[inputn, inputps] = mapminmax(x_train);   %inputn�Ǿ�����һ��������ݣ�inputps�ǹ�һ�������еĲ�����ÿ�������ľ�ֵ����׼�
[outputn, outputps] = mapminmax(y_train);
hiddenLayer = ones(1,nhl);
hiddenLayer = hiddenLayer * 1;   % hiddenLayer��ʾ���ز�Ĳ�����ÿ����Ԫ������
net = feedforwardnet(hiddenLayer,'trainlm');  % Ҳ������net = newff(inputn, outputn, hiddenLayer); 
[net, tr] = train(net, inputn, outputn, 'Useparallel','yes');

inputn_test = mapminmax('apply', x_test, inputps);  %��Ԥ�⼯����ѵ������������Ĳ������й�һ������
an = sim(net, inputn_test, 'Useparallel','yes');   
BPoutput = mapminmax('reverse', an, outputps);    %��Ԥ��������ѵ���������Ӧ�Ĳ����������һ������
BPoutput = round(BPoutput);
ncor = 0;    % number of correct predictions
for i=1:nte
    if isequal(BPoutput(:,i),y_test(:,i))
        ncor = ncor + 1;
    end
end
accuracy_pred = ncor / nte * 100;
fprintf(1,'Ԥ���׼ȷ���ǣ� %4.2f%% \n', accuracy_pred);

an = sim(net, inputn, 'Useparallel','yes');   
BPoutput = mapminmax('reverse', an, outputps);    %��Ԥ��������ѵ���������Ӧ�Ĳ����������һ������
BPoutput = round(BPoutput);
ncor = 0;    % number of correct predictions
for i=1:ntr
    if isequal(BPoutput(:,i),y_train(:,i))
        ncor = ncor + 1;
    end
end
accuracy_return = ncor / ntr * 100;
fprintf(1,'���е�׼ȷ���ǣ� %4.2f%% \n', accuracy_return);
% mesh(x,y,z)