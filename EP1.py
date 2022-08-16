"""
EP1 - Numérico
Alunos:
Gabriel Boaventura Scholl, NUSP: 10771218
William Simões Barbosa, NUSP: 9837646
"""
import matplotlib.pyplot as plt
import numpy as np # Usar apenas para aritmetica de vetores, matrizes, leitura e escrita de dados.
import math

def vetoresAlfaBetaGama(matrizA): #Função que recebe uma matriz e retorna os valores das três diagonais (Alfa, Beta e Gama)
    alfa = []
    beta = []
    gama = []

    for i in range(len(matrizA)): 
        for j in range(len(matrizA[0])):
            if (i == j):
                alfa.append(matrizA[i][j])
            elif (i == j + 1):
                beta.append(matrizA[i][j])
            elif (j == i + 1):
                gama.append(matrizA[i][j])
    return alfa, beta, gama

def rotacaoDeGivens(k, n, ck, sk):
    #Realizando as rotações de Givens para as Qk operações
    Qk = np.identity(n, dtype = float)
    Qk[0][0] = ck
    Qk[0][1] = -sk
    Qk[1][0] = sk
    Qk[1][1] = ck
    
    for i in range(k-1):
        Qk[i+2][i+2] = Qk[i][i]
        Qk[i][i] = 1

        Qk[i+1][i+2] = Qk[i][i+1]
        Qk[i][i+1] = 0

        Qk[i+2][i+1] = Qk[i+1][i]
        Qk[i+1][i] = 0

    return Qk

def decomposicaoQR(matrizA, n):
    #Dada uma matriz de dimensão n, obter sua decomposição Q, R
    I = np.identity(n, dtype = float)
    R = np.copy(matrizA)
    Q = np.copy(I)
    for k in range(1, n): #Transformações Qk, com 1 =< k <= n-1
        alfa, beta, gama = vetoresAlfaBetaGama (R)
        ck = alfa[k-1]/math.sqrt(alfa[k-1]**2 + beta[k-1]**2)
        sk = -beta[k-1]/math.sqrt(alfa[k-1]**2 + beta[k-1]**2)
        Qi = rotacaoDeGivens (k, n, ck, sk) 
        R = Qi @ R
        Q = Q @ Qi.T
    
    return Q, R

def algoritmoQR(A0, V0,  n, erro):
    #Função que recebe a matriz A0 de dimensão n e um erro definido no enunciado, V0 a princípio corresponde à matriz identidade
    A = np.copy(A0)
    V = np.copy(V0)
    alfa, beta, gama = vetoresAlfaBetaGama(A)
    k = 0 #Calcular o número de passos
    while (abs(beta[0]) >= erro):
        Q, R = decomposicaoQR (A, n)
        A = R @ Q
        V = V @ Q
        alfa, beta, gama = vetoresAlfaBetaGama(A)
        k += 1
    print("Passos: ", k)

    return A, V

def reduzDimensao(An):
    #Função responsável por diminuir a dimensão da matriz Ak caso encontre um autovalor.
    n = len(An)
    Anova = np.identity(n-1, dtype = float)
    for i in range(n-1):
        for j in range(n-1):
            Anova[i][j] = An[i][j]
    return Anova

def arrumaDimensao(Qk, nOriginal):
    #Função que recebe a matriz Qk e, como esta matriz pode ter dimensão menor que n na decomposição QR com deslocamento espectral,
    #se certifica que a matriz Q usada para o cálculo de V[k+1] sempre tem dimensão n.
    #Por isso carregamos o valor da dimensão original da matriz A: 'nOriginal'
    Qnova = np.identity(nOriginal, dtype = float)
    for i in range(len(Qk)):
        for j in range(len(Qk[0])):
            Qnova[i][j] = Qk[i][j]
    return Qnova

def inverteLista(lista):
    inversa = []
    for i in range(len(lista)-1,-1,-1):
        inversa.append(lista[i])
    return inversa

def algoritmoQR_deslocamento(A0, V0, n, erro):
    #Função que recebe a matriz A e o erro passado no enunciado, e retorna uma lista de autovalores,
    #Retorna também a matriz de autovetores. Os autovetores estão nas colunas da matriz.
    #A primeira coluna da matriz devolvida corresponde ao primeiro autovetor, associado ao primeiro autovalor da lista de autovalores.

    Ak = np.copy(A0)
    Vk = np.copy(V0)
    alfa, beta, gama = vetoresAlfaBetaGama(Ak)
    listaAutoValores = [] #vou salvar todos os autovalores conforme for encontrando
    mik = 0
    nOriginal = n #Vou salvar a dimensão original da matriz para obter a matriz de autovetores
    m = n
    k = 0 #Calcular o número de passos
    #print("Ak: \n", Ak)
    while (m >= 2): #Vou realizando o processo até a dimensão da matriz ser 2, sempre que beta[n-1] < erro, diminuir a dimensão
        I = np.identity(n, dtype = float)
        if (k > 0):
            #Cálculo de mik pela heurística de Wilkinson
            dk = (alfa[n-2]-alfa[n-1])/2
            if (dk >= 0):
                sgn = 1
            else:
                sgn = -1
            mik = alfa[n-1] + dk - sgn * math.sqrt(dk**2 + beta[n-2]**2)
            
        Qk, Rk = decomposicaoQR (Ak - mik * I, n) #Realização da decomposição QR com a matriz A atual (2 =< dimensão <= n)
        Ak = Rk @ Qk + mik * I #Dados Qk e Rk da decomposição obtida, calcular A[k+1], usando mik
        #print("Ak: \n", Ak)
        Qk = arrumaDimensao(Qk, nOriginal) #Se dimensão de A for menor que n, preciso arrumar a dimensão de Qk para calcular V[k+1]
        Vk = Vk @ Qk #Cálculo de V[k+1] a partir de V[k] e da matriz Q
        alfa, beta, gama = vetoresAlfaBetaGama(Ak) #Obtendo os novos alfa, beta e gama.

        k += 1
        
        if (abs(beta[n-2]) < erro): #Condicional que verifica se o último valor de beta é menor que o erro pedido em enunciado.
            listaAutoValores.append(alfa[n-1]) #Se este último beta for menor, significa que o último alfa é uma aproximação para
            #um autovalor. Então salvo este valor na lista de autovalores que será retornada.
            m -= 1 
            n -= 1 #Caso este último beta seja menor que o erro dado, posso diminuir a dimensão da matriz Ak para aumentar a eficiência
            #do algoritmo. Este detalhe faz o número de passos reduzir consideravelmente.
            Ak = reduzDimensao(Ak) #Reduzindo a dimensão da matriz para continuar buscando os autovalores e autovetores.

        alfa, beta, gama = vetoresAlfaBetaGama(Ak) #Atualizando as listas de alfa, beta e gama para a dimensão n atual
    print("Passos: ", k)
    autovalores = devolveAutovalores(Ak) #Pegando o último autovalor que faltava, todos os demais já estão na lista 'listaAutoValores'
    for i in range(len(listaAutoValores)-1, -1, -1):
        autovalores.append(listaAutoValores[i]) #Aqui simplesmente invertemos os autovalores da lista 'listaAutovalores'
                                                #Dessa forma, a ordem de autovalores corresponde à ordem de autovetores da matriz Vk

    return autovalores, Vk

def devolveAutovalores (matrizA):
    #Função usava para devolver os autovalores da matriz Ak após todas as iterações
    autovalores = []
    for i in range(len(matrizA)):
        autovalores.append(matrizA[i][i])
    return autovalores

def modoMaiorFrequencia(Vk, autovalores): #Função que retorna o modo de maior frequência
    #Para tal, esta função identifica qual é o maior autovalor encontrado
    maiorAutoValor = autovalores[0]
    indice = 0
    for i in range(len(autovalores)):
        if (autovalores[i] > maiorAutoValor):
            maiorAutoValor = autovalores[i]
            indice = i
    
    listaAutovetorAssociado = []
    for i in range(len(Vk)):
        for j in range(len(Vk[0])):
            if (j == indice):
                listaAutovetorAssociado.append(Vk[i][j])

    return listaAutovetorAssociado

# A função ploarGraficos recebe uma matriz "x" da evolução do sistema no tempo, entre outras entradas, e 
# retorna um gráfico disso.
def plotarGraficos(item, x, i, vetorInicial, passos):
    # Aqui, são feitas as colocações de escritas nos gráficos, como o título e outros. 
    eixo = np.array([valor*0.025 for valor in range(passos+1)])
    plt.plot(eixo, x[i])
    if (vetorInicial == 0):
        vetorInicial = "digitado pelo usuário"
    title = f'{item}) Vetor inicial {vetorInicial} - Posição da massa número: {i+1}'
    plt.title(title)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição da massa (m)")
    # name = r'G:\RL Desktop\Numerico\EP1\imgs\\'+str(item)+'.'+str(vetorInicial)+') Massa '+str(i+1)+'.jpg'
    # plt.savefig(name)
    plt.show() # Descomentar isso pra mostrar os gráficos
    plt.clf()

# Essa função plotarConjunto() faz a mesma coisa que a anterior (plotarGraficos()), mas com o gráfico conjunto
# de todas as massas.
def plotarConjuto(item, x, vetorInicial, passos):
    eixo = np.array([valor*0.025 for valor in range(passos+1)])
    plt.plot(eixo, x.T)
    if (vetorInicial == 0):
        vetorInicial = "digitado pelo usuário"
    title = f'{item}) Vetor inicial {vetorInicial} - Gráfico conjunto - Posição das massas'
    plt.title(title)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição da massa (m)")
    # name = r'G:\RL Desktop\Numerico\EP1\imgs\\'+str(item)+'.'+str(vetorInicial)+') Conjunto.jpg'
    # plt.savefig(name)
    plt.show() # Descomentar isso pra mostrar os gráficos
    plt.clf()

def main():
    # Aqui o usuário escolhe o item do tópico 5 do enunciado do EP.
    item = input("Qual item quer verificar ('a', 'b' ou 'c')? ")    
    # Aqui escolhe se é com deslocamento ou não e logo abaixo, a dimensão da matriz.
    desloc = input("Com deslocamento espectral ('s' para Sim, 'n' para Não)? ")

    if (item == 'a'): #Item 'a': testar a execução do algoritmo QR com e sem deslocamento espectral
        n = int(input("Digite a dimensão da matriz (4, 8, 16, 32): "))
        #Para um primeiro teste (o do enunciado), temos as diagonais com um só valor (2 para diagonal principal, -1 para as demais):
        #Mesmo os valores sendo definidos no enunciado, deixamos o usuário escolher o valor que quer verificar.
        valor_diagonal_principal = int(input("Digite o valor da diagonal principal: "))
        valor_subdiagonal = int(input("Digite o valor da subdiagonal: "))
        #Erro pre-definido no enunciado do item 'a'
        erro = 0.000001
        # V0 começa como a matriz identidade n x n.
        V0 = np.identity(n, dtype = float)
        
        #Aqui e no bloco "for" logo abaixo são colocados os valores das diagonais da matriz A0.
        A0 = valor_diagonal_principal * np.identity(n, dtype = float)
        # Para isso, ele percorre a matriz vendo se a posição é uma diagonal secundária e, caso seja,
        # atribui o valor que o usuário entrou nos inputs acima.
        for i in range(n):
            for j in range(n):
                if (i == j + 1):
                    A0[i][j] = valor_subdiagonal
                if (j == i + 1):
                    A0[i][j] = valor_subdiagonal
        

        # Então é aplicado o algoritmo QR, com ou sem deslocamento espectral, conforme escolhido acima (no
        # começo deste "if").
        if (desloc == 's'): #Se com deslocamento espectral
            autovalores, Vk = algoritmoQR_deslocamento(A0, V0, n, erro)
            print("Autovalores: ", autovalores)
            print("Autovetores: \n", Vk)
        # Abaixo é aplicado o caso sem deslocamento. Nos dois casos os resultados são mostrados para o usuário
        elif(desloc == 'n'):
            Ak, Vk = algoritmoQR(A0, V0, n, erro)
            print("Autovalores: ", devolveAutovalores(Ak))
            print("Autovetores: \n", Vk)
        
    # Os itents b) e c) são tratados como uma situação parecida, por isso estão no mesmo elif.
    elif (item == 'b' or item == 'c'):
        # Aqui não se pede o tamanho da matriz, ele é determinado pelo numero de massas.
        # Para um segundo teste (o do enunciado), temos:
        # Como é um sistema massa-mola, as diagonais estão determinadas a partir das constantes
        # elásticas e das massas. Abaixo são definidas para cada caso, b) ou c).

        entrada_valores_X0 = input("Se deseja digitar manualmente 'n' e o vetor X0, digite 's', se não, digite 'n': ")
        numero_de_massas = 0 #inicializando a variável
        X0 = [] #inicializando o vetor inicial de posições

        if item == 'b':
            massa = 2 #kg
            if (entrada_valores_X0 == "s"):
                numero_de_massas = int(input("Digite n: "))
            else:
                numero_de_massas = 5 #portanto são 6 molas (do enunciado)
            constantes_elasticas = [(40+2*i) for i in range(1, numero_de_massas+2, 1)]
        elif item == 'c':
            massa = 2 #kg
            if (entrada_valores_X0 == "s"):
                numero_de_massas = int(input("Digite n: "))
            else:
                numero_de_massas = 10 #portanto são 11 molas (do enunciado)
            constantes_elasticas = [(40+2*(-1)**i) for i in range(1, numero_de_massas+2, 1)]
        
        # Os valores são, então, renomeados para melhor legibilidade das fórmulas.
        k = constantes_elasticas
        n = numero_de_massas
        # Daqui para baixo, são definidos o erro, V0 e A0 da mesma forma que no item a, com uma
        # diferença a ser comentada.
        # Erro usado.
        erro = 0.000001
        V0 = np.identity(n, dtype = float)
        
        A0 = np.identity(n, dtype = float)
        for i in range(n):
            for j in range(n):
                # A diferença se mostra aqui, com as definições das diagonais sendo funções dos "k" e das massas.
                if (i == j):
                    A0[i][j] = (k[i]+k[i+1])/massa
                if (i == j + 1):
                    A0[i][j] = -k[i]/massa
                if (j == i + 1):
                    A0[i][j] = -k[j]/massa

        if (desloc == 's'):
            #É aplicado o algoritmo QR com deslocamento espectral, conforme pedido e os valores são impressos.
            autovalores, Vk = algoritmoQR_deslocamento(A0, V0, n, erro)
            print("Com deslocamento:")
            print("Autovalores (Frequências de vibração - Hz): \n", autovalores)
            print("Autovetores (Modos naturais de vibração): \n", Vk)
            print()
        elif (desloc == 'n'):
            #É aplicado o algoritmo QR sem deslocamento espectral
            Ak, Vk = algoritmoQR(A0, V0, n, erro)
            autovalores = devolveAutovalores(Ak)
            print("Sem deslocamento:")
            print("Autovalores (Frequências de vibração - Hz): ", autovalores)
            print("Autovetores (Modos naturais de vibração): \n", Vk)

        # Daqui em diante, começa uma escolha de qual vetor X(0), de posições iniciais, o usuário quer.
        # É feito uma pergunta em forma de input e, a partir disso, o vetor é determinado, seja relativo
        # ao item b) ou ao item c).
        if item == 'b':
            vetorInicial = 0
            if (entrada_valores_X0 == 's'): #Usuário digitando X0
                for i in range(n):
                    valor = 0 #inicializando uma variável auxiliar
                    if (i == 0):
                        valor = float(input("Digite o primeiro valor de X0: "))
                    elif (i == n-1):
                        valor = float(input("Digite o último valor de X0: "))
                    else:
                        valor = float(input("Digite o próximo valor de X0: "))
                    X0.append(valor)
                X0 = np.array(X0)

            elif (entrada_valores_X0 == 'n'):
                print("1 - X(0) = -2, -3, -1, -3, -1")
                print("2 - X(0) = 1, 10, -4, 3, -2")
                print("3 - X(0) correspondente ao modo de maior frequência")
                vetorInicial = int(input("Escolha o X0 digitando 1, 2 ou 3: "))
                if (vetorInicial == 1):
                    X0 = np.array([-2, -3, -1, -3, -1])
                elif (vetorInicial == 2):
                    X0 = np.array([1, 10, -4, 3, -2])
                elif (vetorInicial == 3):
                    X0 = np.array(modoMaiorFrequencia(Vk, autovalores))

        elif item == 'c':
            vetorInicial = 0
            if (entrada_valores_X0 == 's'): #Usuário digitando X0
                for i in range(n):
                    valor = 0 #inicializando uma variável auxiliar
                    if (i == 0):
                        valor = float(input("Digite o primeiro valor de X0: "))
                    elif (i == n-1):
                        valor = float(input("Digite o último valor de X0: "))
                    else:
                        valor = float(input("Digite o próximo valor de X0: "))
                    X0.append(valor)
                X0 = np.array(X0)

            elif (entrada_valores_X0 == 'n'):
                print("1 - X(0) = -2, -3, -1, -3, -1, -2, -3, -1, -3, -1")
                print("2 - X(0) = 1, 10, -4, 3, -2, 1, 10, -4, 3, -2")
                print("3 - X(0) correspondente ao modo de maior frequência")
                vetorInicial = int(input("Escolha o X0 digitando 1, 2, ou 3: "))
                if (vetorInicial == 1):
                    X0 = np.array([-2, -3, -1, -3, -1, -2, -3, -1, -3, -1])
                elif (vetorInicial == 2):
                    X0 = np.array([1, 10, -4, 3, -2, 1, 10, -4, 3, -2])
                elif (vetorInicial == 3):
                    X0 = np.array(modoMaiorFrequencia(Vk, autovalores))

        # Com o valor das condições iniciais escolhido, agora prosseguimos com a simulação do sistema
        # massa-mola em intervalos de 0.025 segundos, que são 400 passos em 10 segundos.
        passos = 400 #Plotando o gráfico para 10s, teremos um valor a cada 0.025s
        x = np.zeros((n,passos+1), dtype = float)
        
        # A multiplicação de matrizes abaixo é feita com a notação do numpy. Isso foi considerado como
        # "aritmética de matrizes", permitida pelo enunciado.
        a = Vk.T @ X0.T
        # No loop "for" abaixo, varremos a matriz x, que vai ser a simulação ao longo de um eixo e as
        # massas ao longo de outro, calculando e colocando cada posição de cada massa ao longo do tempo.
        for i in range(n):
            for t in range(passos+1):
                for j in range(n):
                    # O valor de ômega (representado por "w" aqui) é a raiz dos lâmbdas, que estão em autovalores[j].
                    w = math.sqrt(float(autovalores[j]))
                    x[i][t] += Vk[i][j] * a[j] * math.cos(w*t*0.025)

            # Isso aqui embaixo plota graficos individuais pra arquivos de imagem na pasta determinada 
            # dentro da função plotarGraficos, ou apenas mostra para o usuário, dependendo da necessidade.
            plotarGraficos(item,x,i,vetorInicial,passos)

        # Esse é o plot do gráfico de todas as massas em conjunto, da mesma forma que a plotarGraficos().
        plotarConjuto(item,x,vetorInicial,passos)

# Finalmente, com tudo organizado em funções, a main() é chamada para executar o código.
main()
