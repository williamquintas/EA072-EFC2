%rotina que faz a poda de neur�nios que nunca vencem
%par�metros: W - matriz de pesos - dimens�o Nxneuronios
%            wins - vetor que contabiliza as vit�rias (1xneuronios)
%            neuronios - n�mero de neur�nios no mapa
%            Index - matriz de vizinhan�a dos neur�nios

function [W,wins,neuronios,Index] = poda(W,wins,neuronios,Index)

%para todos os neur�nios do mapa
te = neuronios;
while te >= 1
    %checa se o neur�nio nunca venceu
    if(wins(te) == 0)
        W(:,te) = []; %mata coluna correspondente ao neur�nio
        wins(:,te) = []; %elimina coluna correspondente ao neur�nio
        neuronios = neuronios - 1; %reduz o n�mero de neur�nios no mapa
        %ajuste da vizinhan�a
        Index(1,2)=neuronios; Index(neuronios,1)=1; Index(neuronios+1,:)=[];
    end
    te = te - 1;
end