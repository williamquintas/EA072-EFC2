%rotina que inicializa os pesos - arranjo unidimensional em anel
%par�metros: X - matriz dos dados de entrada - NxM

%Esta inicializa��o visa minimizar a viola��o topol�gica na constru��o do
%mapa, ou seja, esperamos que ap�s a auto-organiza��o do mapa, neur�nios
%vizinhos no mapa tenham vetores de pesos pr�ximos no espa�o dos dados

function W = inicializa_pesos(X, neuronios)

%N = n�mero de componentes e M = n�mero de dados (padr�es) de entrada
[N,M] = size(X);
%ponto m�dio dos dados - centro do anel
medio = mean(X,2)'; 
%limitantes dos padr�es de entrada
limite_max=max(max(X));
limite_min=min(min(X));
%for�amos os dados de entrada terem m�dia zero
Xmz = X - repmat(medio',1,M);
%matriz de autocorrela��o - PCA
R = Xmz*Xmz';
%autovalores e autovetores
[V,sem_uso] = eig(R);
%dois maiores autovalores definem as duas componentes principais
direcao = [V(:,N-1) V(:,N)];
%raio do anel
raio = abs(0.2*(limite_max - limite_min));
%�ngulo de varia��o
angles = linspace(0,2*pi,neuronios);
%montamos a matriz dos pesos dos neur�nios - um anel no espa�o dos dados
W = cos(angles)'*raio*direcao(:,1)'+sin(angles)'*raio*direcao(:,2)'+repmat(medio',1,neuronios)';
W = W'; %manter coer�ncia entre as rotinas
