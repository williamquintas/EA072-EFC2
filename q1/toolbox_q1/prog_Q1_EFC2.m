% Programa vinculado � Q1 do EFC2
% EA072 - Intelig�ncia Artificial em Aplica��es Industriais
% FEEC/Unicamp
% Prof. Fernando J. Von Zuben
% 30/10/2019
clear all;
close all;
% Define o tamanho da popula��o [tam_pop], como um n�mero par
tam_pop = 200;
% Define as taxas de crossover e de muta��o
pc = 0.5;
pm = 0.7;
% Define o intervalo de excurs�o das vari�veis
v_inf = 0.0;
v_sup = 20.0;
% Define o n�mero de gera��es
n_ger = 50;
% Estruturas para preservar o melhor indiv�duo a cada gera��o e seu fitness
v_melhor_fitness = [];
v_fitness_medio = [];
v_melhor_ind = [];
% Gera a popula��o inicial respeitando o intervalo de excurs�o das vari�veis. Cada indiv�duo � uma linha da matriz [pop].
pop = v_inf+(v_sup-v_inf)*rand(tam_pop,3);
% Avalia a popula��o inicial
for i=1:tam_pop,
    kp = pop(i,1);
    kd = pop(i,2);
    ki = pop(i,3);
    S = stepinfo_Q1(G_MF(kp,kd,ki));
    [Gm,Pm,Wcg,Wcp] = margin(G_MF(kp,kd,ki));
    t1 = 0.0;t2 = 0.0;t3 = 0.0;
    if isnan(S.SettlingTime) | isnan(S.RiseTime) | Pm <= 0,
        fitness(i,1) = 0;;
    else
        if S.SettlingTime > 0.5,
            t1 = S.SettlingTime - 0.5;
        end
        if S.RiseTime > 0.04,
            t2 = S.RiseTime - 0.04;
        end
        t3 = (Pm-60)^2;
        fitness(i,1) = 1/(t1+(0.5/0.04)*t2+t3+1);
    end
end
% Loop para gerar a nova popula��o at� atingir um n�mero m�ximo de gera��es
[melhor_fitness,melhor_ind] = max(fitness);
v_melhor_fitness = [v_melhor_fitness;melhor_fitness];
v_fitness_medio = [v_fitness_medio;mean(fitness)];
v_melhor_ind = [v_melhor_ind;pop(melhor_ind,:)];
v_delta = [];
for k = 1:n_ger,
    % Em lugar da roleta, adota-se torneio de 3, pois deve-se excluir os
    % indiv�duos de fitness nulo e a implementa��o fica melhor
    candidatos = [];
    for i=1:tam_pop,
        if fitness(i,1) > 0,
            candidatos = [candidatos;i];
        end
    end
    n_tam_pop = length(candidatos);
    n_pop = [];
%    [[1:tam_pop]' pop fitness]
    for i=1:tam_pop,
        v_aux = randperm(n_tam_pop)';
        torneio = v_aux(1:3,1);
        c_fitness(1,1) = fitness(candidatos(torneio(1,1),1),1);
        c_fitness(2,1) = fitness(candidatos(torneio(2,1),1),1);
        c_fitness(3,1) = fitness(candidatos(torneio(3,1),1),1);
        [v_max,ind_max] = max(c_fitness);
        n_pop = [n_pop;pop(candidatos(torneio(ind_max,1),1),:)];
    end
    % Aplica��o de crossover onde for escolhido
    for j=1:(tam_pop/2),
        if rand(1,1) <= pc,
            % 50% de chance de ocorrer o crossover aritm�tico
            if rand(1,1) <= 0.5,
                a = -0.1+1.2*rand(1,1);
                n_pop1 = a*n_pop(2*(j-1)+1,:)+(1-a)*n_pop(2*(j-1)+2,:);
                n_pop2 = (1-a)*n_pop(2*(j-1)+1,:)+a*n_pop(2*(j-1)+2,:);
                n_pop(2*(j-1)+1,:) = n_pop1;
                n_pop(2*(j-1)+2,:) = n_pop2;
            else
                % 50% de chance de ocorrer o crossover uniforme
                for z=1:3,
                    if rand(1,1) <= 0.5;
                        n_pop1(1,z) = n_pop(2*(j-1)+1,z);
                        n_pop2(1,z) = n_pop(2*(j-1)+2,z);
                    else
                        n_pop1(1,z) = n_pop(2*(j-1)+2,z);
                        n_pop2(1,z) = n_pop(2*(j-1)+1,z);
                    end
                end
                n_pop(2*(j-1)+1,:) = n_pop1;
                n_pop(2*(j-1)+2,:) = n_pop2;
            end
        end
    end
    % Aplica��o de muta��o n�o-uniforme onde for escolhido
    n_mut = round(tam_pop*3*pm);
    v_aux = randperm(tam_pop*3)';
    bits_mutados = v_aux(1:n_mut,1);
    for i=1:n_mut,
        if rem(bits_mutados(i),3) == 0,
            linha = fix(bits_mutados(i)/3);
            coluna = 3;
        else
            linha = fix(bits_mutados(i)/3)+1;
            coluna = rem(bits_mutados(i),3);
        end
        if rand(1,1) <= 0.5,
            delta = mut_nunif(k,v_sup-n_pop(linha,coluna),n_ger);
            v_delta = [v_delta;delta];
            n_pop(linha,coluna) = n_pop(linha,coluna) + delta;
        else
            delta = mut_nunif(k,n_pop(linha,coluna)-v_inf,n_ger);
            v_delta = [v_delta;delta];
            n_pop(linha,coluna) = n_pop(linha,coluna) - delta;
        end
    end
    % Avalia��o da nova popula��o
    for i=1:tam_pop,
        kp = n_pop(i,1);
        kd = n_pop(i,2);
        ki = n_pop(i,3);
        S = stepinfo_Q1(G_MF(kp,kd,ki));
        [Gm,Pm,Wcg,Wcp] = margin(G_MF(kp,kd,ki));
        t1 = 0.0;t2 = 0.0;t3 = 0.0;
        if isnan(S.SettlingTime) | isnan(S.RiseTime) | Pm <= 0,
            fitness(i,1) = 0;
        else
            if S.SettlingTime > 0.5,
                t1 = S.SettlingTime - 0.5;
            end
            if S.RiseTime > 0.04,
                t2 = S.RiseTime - 0.04;
            end
            t3 = (Pm-60)^2;
            fitness(i,1) = 1/(t1+(0.5/0.04)*t2+t3+1);
        end
    end
    % Preserva��o do melhor indiv�duo da gera��o anterior, se melhor que o melhor da nova gera��o
    melhor_fitness1 = melhor_fitness;
    melhor_ind1 = melhor_ind;
    [melhor_fitness,melhor_ind] = max(fitness);
    if melhor_fitness1 > melhor_fitness,
        [pior_fitness,pior_ind] = min(fitness);
        n_pop(pior_ind,:) = pop(melhor_ind1,:);
        fitness(pior_ind,1) = melhor_fitness1;
        melhor_fitness = melhor_fitness1;
        melhor_ind = pior_ind;
    end
    pop = n_pop;
    v_melhor_fitness = [v_melhor_fitness;melhor_fitness];
    v_fitness_medio = [v_fitness_medio;mean(fitness)];
    v_melhor_ind = [v_melhor_ind;pop(melhor_ind,:)];
    kp = pop(melhor_ind,1);
    kd = pop(melhor_ind,2);
    ki = pop(melhor_ind,3);
    figure(1);step(G_MF(kp,kd,ki));drawnow;
    S = stepinfo_Q1(G_MF(kp,kd,ki));
    [Gm,Pm,Wcg,Wcp] = margin(G_MF(kp,kd,ki));
    disp(sprintf('Gera��o %d: T_sub = %g | T_acom = %g | Sobrs = %g | Margfase = %g',k,S.RiseTime,S.SettlingTime,S.Overshoot,Pm));
    disp(sprintf('Gera��o %d: Os melhores valores encontrados foram: k_p = %g; k_d = %g; k_i = %g',k,kp,kd,ki));
    disp(sprintf('Gera��o %d: Fitness deste melhor indiv�duo = %g',k,melhor_fitness));
end
figure(2);plot(v_melhor_fitness,'k');hold on;plot(v_fitness_medio,'r');hold off;
title('Melhor fitness (preto) e fitness m�dio (vermelho) da popula��o ao longo das gera��es');
figure(3);plot(v_melhor_ind(:,1),'k');hold on;plot(v_melhor_ind(:,2),'r');plot(v_melhor_ind(:,3),'b');hold off;
title('Evolu��o dos ganhos do controlador PID: k_p (preto) | k_d (vermelho) | k_i (azul)');
xlabel('N�mero de gera��es');
figure(4);plot(v_delta);
title('Intensidade da muta��o n�o-uniforme ao longo das gera��es');
