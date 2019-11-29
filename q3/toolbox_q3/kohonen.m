%treinamento de um mapa de Kohonen - arranjo unidimensional - anel
%mecanismos de poda e inser��o autom�tica de neur�nios
%matriz X - colunas representam os padr�es de entrada(N atributos x M padr�es)

function [W,Index,neuronios] = kohonen(X,neuronios,gama_inicial,radius_inicial,LIMIAR,LIMITE_TAXA,PERIODO,MAX_ITERATION)

%N = n�mero de componentes e M = n�mero de dados (padr�es) de entrada
[sem_uso,M] = size(X);

%n�mero de itera��es
iteration = 1;
%taxa de aprendizado
gama = gama_inicial; 
%radius definir� a vizinhan�a - no in�cio, ambos vizinhos s�o afetados
%conforme o n�mero de itera��es aumenta, n�o teremos mais o ajuste
radius = radius_inicial;

%matriz com o �ndice dos vizinhos topol�gicos - anel
vetor = (1:neuronios)';
Index = [circshift(vetor,-1) circshift(vetor,1)];
%vetor que contabiliza as vit�rias dos neur�nios
wins = zeros(1,neuronios);

%inicializa��o da matriz de pesos
W = inicializa_pesos(X,neuronios);
%exibe configura��o inicial - pesos e dados
figure(1); plot_SOM(X,W,neuronios);
title('Configuração inicial dos neurônios(o)');

%contador do n�mero de neur�nios pr�ximos aos padr�es
cont = 0; 

%crit�rio de parada:
%   n�mero m�ximo de �pocas
%   contador do n�mero de neur�nios que se encontram muito pr�ximos aos
%   dados - ou seja, cuja dist. ao dado (cidade) mais pr�ximo � inferior ao
%   LIMIAR
%   taxa de aprendizado maior que um valor m�nimo permitido

while iteration < MAX_ITERATION && cont < M && gama > LIMITE_TAXA
    
    %a cada PERIODO itera��es, zeramos o contador de vit�rias
    if(mod(iteration,PERIODO)==0)
        wins = zeros(1,neuronios);
    end
    
    %ordena aleatoriamente os padr�es de entrada
    X = X(:,randperm(M));
    %contador do n�mero de neur�nios pr�ximos aos padr�es
    cont = 0;
   
    %apresenta os M padroes
    for i=1:1:M
        %determinamos o neur�nio vencedor
        [indice, value] = vencedor(X(:,i),W);
        %contabiliza uma vit�ria para o neur�nio "indice"
        wins(indice) = wins(indice) + 1;
        if value < LIMIAR
           %se todos os neur�nios est�o com proximidade menor que um LIMIAR
           cont = cont + 1;
        end
        %ajustamos os pesos do vencedor e dos neur�nios vizinhos
        W = ajuste_peso(W, X, Index, indice, gama, i, radius);
    end
    
    %ajusta a taxa de aprendizado gama e a vizinhan�a
    gama = gama_inicial*exp(-(iteration)/MAX_ITERATION);  %const de tempo
    radius = radius_inicial*exp(-(iteration)/MAX_ITERATION);
    
    %a cada PERIODO itera��es, fazemos as opera��es de poda e inser��o de neur�nios
    if(mod(iteration,PERIODO)==0)
        %etapa de poda - neur�nios que nunca vencem
        [W,wins,neuronios,Index] = poda(W,wins,neuronios,Index);
        %etapa de inser��o - pr�ximo a neur�nios que vencem muito
        [W,wins,neuronios,Index] = insere(W,wins,neuronios,Index);
    end
    %exibi��o dos par�metros
    fprintf('Iteracao:%d \t Taxa (Gama):%1.4f \t Raio:%d \t N:%d\n',iteration,gama,round(radius),neuronios);
    figure(2); plot_SOM(X,W,neuronios); title(['Configuração dos neurônios - iteração ' num2str(iteration)]); drawnow;
    iteration = iteration + 1;
end