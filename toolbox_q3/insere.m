%rotina que faz a poda de neur�nios que nunca vencem
%par�metros: W - matriz de pesos - dimens�o Nxneuronios
%            wins - vetor que contabiliza as vit�rias (1xneuronios)
%            neuronios - n�mero de neur�nios no mapa
%            Index - matriz de vizinhan�a dos neur�nios

function [W,wins,neuronios,Index] = insere(W,wins,neuronios,Index)

%n�mero m�ximo de vit�rias
max_wins = max(wins);
%porcentagem do n�mero m�ximo de vit�rias
alpha = 0.5; %controla a qtde de neur�nios q � adicionada
te = 1;
%para todos os neur�nios do mapa
while te <= neuronios
    %se o neur�nio obteve o limiar de vit�rias
    if wins(te) > 1 && wins(te) >= alpha*max_wins
        %adiciona coluna no vetor de vit�rias
        wins = [wins(:,1:te) (wins(:,te)-1) wins(:,te+1:neuronios)];
        %adiciona coluna na matriz de pesos
        W = [W(:,1:te) (0.02*rand-0.01+W(:,te)) W(:,te+1:neuronios)];
        %aumentamos o n�mero de neur�nios
        neuronios = neuronios + 1;
        %aumentamos o �ndice para pular o neur�nio inserido agora
        te = te + 1;
        %ajuste da vizinhan�a
        Index(1,2)=neuronios; Index(neuronios-1,1)=neuronios; Index(neuronios,:)=[1 neuronios-1];
    end
    %pr�ximo neur�nio
    te = te + 1;
end