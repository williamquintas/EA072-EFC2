%Toolbox - Mapa de Kohonen unidimensional aplicado ao problema do caixeiro viajante
%Especifica��es:
%   vizinhan�a - unidimensional em anel
%   n�mero de neur�nios - controlado dinamicamente por mecanismos de poda e inser��o
%   inser��o - 
%   poda - 

clear; close all;

%carrega as coordenadas das cidades
load dados.mat

%seleciona a inst�ncia do caixeiro viajante
choice = menu('Escolha a instância do caixeiro viajante','Berlin52','Inst. 1 (100 cidades) ','Inst. 2 (100 cidades)');
if choice == 1
    load berlin52
elseif choice == 2
    load inst100x100.mat
else
    load dados.mat;
end

%par�metros do SOM

%n�mero inicial de neur�nios
N = 10;
%n�mero m�ximo de �pocas do processo de auto-organiza��o
max_epoch = 500;
%n�mero de �pocas para zerar contador de vit�rias e realizar etapas de poda e inser��o
PERIODO = 5;
%limiar de proximidade do neur�nio vencedor ao padr�o
limiar = 0.01;
%taxa de aprendizado inicial
gama = 0.2;
%limiar (valor m�nimo permitido) da taxa de aprendizado
limite_taxa = 0.01;
%raio (extens�o) da vizinhan�a - no in�cio, os vizinhos (esquerdo e direito) s�o afetados de forma mais intensa; conforme o n�mero de �pocas aumenta, n�o teremos mais o ajuste
radius = 1;

%mapa auto-organiz�vel de Kohonen
[W,Index,Nf] = kohonen(X,N,gama,radius,limiar,limite_taxa,PERIODO,max_epoch);

%exibi��o de resultados
close(2); figure; plot_SOM(X,W,Nf);

%determinamos o percurso a ser realizado

%matriz com as dist�ncias entre cada neur�nio e cidade
mt = dist(W',X);
%matriz com as dist�ncias entre as cidades
mx = dist(X', X); 
solucao = zeros(1,Nf); d = 0;
%encontramos a primeira cidade
[sem_uso,ind] = min(mt(1,:));
solucao(1) = ind;
for k=1:Nf-1
    %ind corresponde ao �ndice da cidade representada pelo k-�simo neur�nio
    [sem_uso,ind] = min(mt(k+1,:));
    %montamos o percurso com a ordem das cidades
    solucao(k+1) = ind;
    %dist�ncia entre as cidades � acumulada
    d = d + mx(solucao(k),solucao(k+1));
end
%inclu�mos tamb�m a dist�ncia entre a 1a e a �ltima cidade
d = d + mx(solucao(1),solucao(Nf));
title(['Configuração final dos neurônios(o) - Percurso total: ' num2str(d)]);
