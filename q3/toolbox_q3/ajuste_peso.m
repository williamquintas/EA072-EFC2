%rotina que ajusta os pesos do neur�nio vencedor e de seus vizinhos
function W = ajuste_peso(W, X, Index, indice, gama, i_padrao, radius)

%vari�veis auxiliares
padrao = X(:,i_padrao); %vetor com o padr�o apresentado
ind_sup = Index(indice,2); %�ndice do vizinho superior
ind_inf = Index(indice,1); %�ndice do vizinho inferior
j = round(radius); %simula a redu��o na vizinhan�a
if j > 0
    %ajuste do vetor de pesos do vizinho superior
    W(:,ind_sup) = W(:,ind_sup) + gaussmf(indice + 1,[radius,indice])*gama*(padrao - W(:,ind_sup));
    %ajuste do vetor de pesos do vizinho inferior
    W(:,ind_inf) = W(:,ind_inf) + gaussmf(indice - 1,[radius,indice])*gama*(padrao - W(:,ind_inf));
end
%ajuste do vetor de pesos do neur�nio vencedor
W(:,indice) = W(:,indice) + gama*(X(:,i_padrao) - W(:,indice));