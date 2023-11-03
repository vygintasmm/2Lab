x = 0.1:1/22:1; % Įėjimo vektorius
y = (1 + 0.6*sin(2*pi*x/0.7)) + 0.3*sin(2*pi*x)/2; % Norimas atsakymas
hiddenLayerSize = 6; % Pasirenkame 6 neuronus paslėptajame sluoksnyje
net = feedforwardnet(hiddenLayerSize); 
net.divideParam.trainRatio = 70/100; % Mokymo duomenų rinkinio dalis
net.divideParam.valRatio = 15/100;   % Tikrinimo duomenų rinkinio dalis
net.divideParam.testRatio = 15/100;  % Testavimo duomenų rinkinio dalis
[net,tr] = train(net,x,y);
outputs = net(x);
errors = gsubtract(outputs,y); % Skirtumas tarp gauto ir tikrojo atsakymo
performance = perform(net,y,outputs); % Mokymo efektyvumo įvertinimas
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, plotfit(outputs, y)

