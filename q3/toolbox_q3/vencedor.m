%rotina que devolve o �ndice do neur�nio vencedor
%padrao = vetor coluna dos atributos da entrada
%W = matriz com os pesos associados a cada neur�nio

function [indice, value] = vencedor(padrao,W)

%monta vetor com a dist�ncia entre os neur�nios e a entrada
d = dist(padrao',W);
%determina o �ndice do neur�nio vencedor
[value,indice] = min(d);
