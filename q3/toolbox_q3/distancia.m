%rotina que calcula a distancia euclidiana entre dois indiv�duos

function dist = distancia(aux, aux1)

%distancia euclidiana entre os indiv�duos
dist = sqrt(sum((aux-aux1).^2));
