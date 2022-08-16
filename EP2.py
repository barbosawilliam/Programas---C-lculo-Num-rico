"""
EP2 - Numérico
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

def arrumaDimensao(Qk, nOriginal):
    #Função que recebe a matriz Qk e, como esta matriz pode ter dimensão menor que n na decomposição QR com deslocamento espectral,
    #se certifica que a matriz Q usada para o cálculo de V[k+1] sempre tem dimensão n.
    #Por isso carregamos o valor da dimensão original da matriz A: 'nOriginal'
    Qnova = np.identity(nOriginal, dtype = float)
    for i in range(len(Qk)):
        for j in range(len(Qk[0])):
            Qnova[i][j] = Qk[i][j]
    return Qnova

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

def reduzDimensao(An):
    #Função responsável por diminuir a dimensão da matriz Ak caso encontre um autovalor.
    n = len(An)
    Anova = np.identity(n-1, dtype = float)
    for i in range(n-1):
        for j in range(n-1):
            Anova[i][j] = An[i][j]
    return Anova

def devolveAutovalores (matrizA):
    #Função usava para devolver os autovalores da matriz Ak após todas as iterações
    autovalores = []
    for i in range(len(matrizA)):
        autovalores.append(matrizA[i][i])
    return autovalores

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
    #print("Passos: ", k)
    autovalores = devolveAutovalores(Ak) #Pegando o último autovalor que faltava, todos os demais já estão na lista 'listaAutoValores'
    for i in range(len(listaAutoValores)-1, -1, -1):
        autovalores.append(listaAutoValores[i]) #Aqui simplesmente invertemos os autovalores da lista 'listaAutovalores'
                                                #Dessa forma, a ordem de autovalores corresponde à ordem de autovetores da matriz Vk

    return autovalores, Vk

def devolve_e(d): #Devolve o vetor 'e' de dimensão 'd'
	e = [1]
	for i in range(d-1):
		e.append(0)
	return e

def norma(vetor):
	# Calcula a norma de um vetor.
	norma = 0
	for el in vetor:
		norma += el**2
	return math.sqrt(norma)

def devolve_wi(matrizA, n, i): #Dada a matriz A, esta função retorna o vetor wi
	wi = []
	ai = []
	for linha in range(i, n): #Se i for igual a 1, vou calcular w1, e pra isso preciso do vetor a1, que está na coluna 0 da matriz A 
		ai.append(matrizA[linha][i-1])
	#print(ai)
	delta = 1
	if (ai[0] < 0):
		delta = delta * (-1)
	e = devolve_e(n-i)

	for it in range(len(ai)): #it = iterador
		wi.append(ai[it] + delta*e[it]*norma(ai))
	#print("w%d: "%i, wi)
	return wi

def devolve_aj(matrizA, n, i, j):
	aj = []
	for coluna in range(i, n):
		aj.append(matrizA[j][coluna])
	return aj

def devolve_ai(matrizA, n, i, j): #Dada a matriz A, esta função retorna o vetor ai da coluna j, de dimensão n - i
	ai = []
	for linha in range(i, n):
		ai.append(matrizA[linha][j])
	return ai

def somaVetores (vetorA, vetorB): #Retorna a soma de dois vetores A e B quaisquer
	vetorSoma = []
	for it in range(len(vetorA)):
		vetorSoma.append(vetorA[it]+vetorB[it])
	return vetorSoma

def produtoPorEscalar (escalar, vetor): #Retorna o produto de um vetor A qualquer por um escalar
	novoVetor = []
	for it in range(len(vetor)):
		novoVetor.append(escalar*vetor[it])
	return novoVetor

def produtoEscalar (vetorA, vetorB): #Retorna o produto escalar entre dois vetores A e B quaisquer
	escalar = 0
	for it in range(len(vetorA)):
		escalar += vetorA[it]*vetorB[it]
	return escalar

def aplicaColunaNaMatriz(matriz, resultadoColuna, i, j): #Substituindo a coluna calculado na matriz
	for linha in range(i, len(matriz)):
		matriz[linha][j] = resultadoColuna[linha-i]

def aplicaLinhaNaMatriz(matriz, resultadoLinha, i, j):
	for coluna in range(i, len(matriz)):
		matriz[j][coluna] = resultadoLinha[coluna-i]

def multiplica_H_esquerda (matriz, n, wi, i): #Primeiro acha o valor de ai (vetor de cada coluna i da matriz)
	#Em posse do vetor ai, realiza a transformação de Householder de ai no vetor wi
	for j in range(i-1, n):
		aj = devolve_ai(matriz, n, i, j)
		#print("a%d"%j, aj)
		escalarWX = produtoEscalar(wi, aj)
		escalarWW = produtoEscalar(wi, wi)
		coeficiente = -2 * (escalarWX/escalarWW)
		multiploW = produtoPorEscalar(coeficiente, wi)
		resultadoColuna = somaVetores(aj, multiploW)
		aplicaColunaNaMatriz(matriz, resultadoColuna, i, j)

def multiplica_H_direita (matriz, n, wi, i): #Primeiro acha o valor de aj (vetor de cada linha j da matriz, exceto a 1a)
	#Com o vetor aj em mãos, aplicar a transformação de Householder de aj no vetor wi
	#Nesta tranformação, não há a necessidade de pegar a1 (vetor da primeira linha), pois a matriz é simétrica
	for it in range(i, n):
		matriz[i-1][it] = matriz[it][i-1] #Pra não aplicar a transformação na primeira linha, pois a matriz é simétrica

	for j in range(i, n):
		aj = devolve_aj(matriz, n, i, j)
		#print("a%d"%j, aj)
		escalarWX = produtoEscalar(wi, aj)
		escalarWW = produtoEscalar(wi, wi)
		coeficiente = -2 * (escalarWX/escalarWW)
		multiploW = produtoPorEscalar(coeficiente, wi)
		resultadoLinha = somaVetores(aj, multiploW)
		aplicaLinhaNaMatriz(matriz, resultadoLinha, i, j)

def achaH_transposto (matriz, n, wi, i): #Aplicando a transformação de householder para achar a matriz H transposta

	for j in range(n):
		aj = devolve_aj(matriz, n, i, j)
		#print("a%d"%j, aj)
		escalarWX = produtoEscalar(wi, aj)
		escalarWW = produtoEscalar(wi, wi)
		coeficiente = -2 * (escalarWX/escalarWW)
		multiploW = produtoPorEscalar(coeficiente, wi)
		resultadoLinha = somaVetores(aj, multiploW)
		aplicaLinhaNaMatriz(matriz, resultadoLinha, i, j)

def devolve_TridiagonalSimetrica(matrizA, n): #Aplicando a transformação de Householder
	matrizT = np.copy(matrizA)
	HT = np.identity(n, dtype = float) #Inicializando H transposto como a identidade

	for i in range(1, n-1): #Número de etapas até obtermos a matriz tridiagonal simétrica (n-2)
		wi = devolve_wi(matrizT, n, i)
		#print("w%d: "%i, wi)
		multiplica_H_esquerda (matrizT, n, wi, i)
		#print(matrizT)
		multiplica_H_direita (matrizT, n, wi, i)
		#print(matrizT)
		achaH_transposto (HT, n, wi, i)
		#print(I)

	return matrizT, HT

def devolve_autovetores(matriz, coluna):
	lista_autovetores = []
	for i in range(len(matriz)):
		lista_autovetores.append(matriz[i][coluna])
	return lista_autovetores

def rotinaDeFormacao(barra, K, M, ro, area, elasticidade):
	# Esta função coloca as contribuições nas matrizes K e M.
	# Primeiro definimos os valores.
	# print(f'barra: {barra}')
	i = barra[0]
	j = barra[1]
	angulo = barra[2]
	angulorad = math.radians(angulo)
	comprimento = barra[3]
	c = math.cos(angulorad)
	s = math.sin(angulorad)
	senosEcossenos = np.array([[c**2,c*s,-c**2,-c*s],
							   [c*s,s**2,-c*s,-s**2],
							   [-c**2,-c*s,c**2,c*s],
							   [-c*s,-s**2,c*s,s**2]])

	# Então adicionamos as contribuições nas posições correspondentes (matriz de rigidez total).
	index1 = -1
	index2 = -1
	for a in [2*i-1, 2*i, 2*j-1, 2*j]:
		index1 += 1
		index2 = -1
		for b in [2*i-1, 2*i, 2*j-1, 2*j]:
			index2 += 1
			a, b = int(a), int(b)
			# print(f'Calculating K[{a-1}][{b-1}]')
			if a < 25 and b < 25:
				K[a-1][b-1] += (area*elasticidade/comprimento)*(senosEcossenos[index1][index2])
			else:
				pass
			
	# Adiciona as contribuições às massas dos nós extremos da barra.
	for (c,d) in [(2*i-1,2*i-1),(2*i,2*i),(2*j-1,2*j-1),(2*j,2*j)]:
		c, d = int(c), int(d)
		# print(f'Calculating M[{c-1}][{d-1}]')
		if c < 25 and d < 25:
			M[c-1][d-1] += ro*comprimento*area/2
	return K, M

def matrixInverter(M):
	# Só funciona se a matriz for diagonal, que é o caso da matriz M de massas.
	M_inv = np.zeros_like(M)
	for i in range(len(M)):
		for j in range(len(M[0])):
			if (M[i][j] != 0):
				M_inv[i][j] = 1/M[i][j]
	
	return M_inv

def devolveIndices(frequencias):
	cincoMenoresFrequencias = []
	ordenaFrequencias = [] #criando uma cópia para não bagunçar os indices de 'frequencias', tbm estou salvando os índices
	for i in range(len(frequencias)):
		ordenaFrequencias.append([frequencias[i], i])
		#ordenaFrequencias.append(i)

	for i in range(len(frequencias)):
		for j in range(i, len(frequencias)):
			if (ordenaFrequencias[j][0] < ordenaFrequencias[i][0]):
				tempValor = ordenaFrequencias[j][0]
				tempIndice = ordenaFrequencias[j][1]
				ordenaFrequencias[j][0] = ordenaFrequencias[i][0]
				ordenaFrequencias[j][1] = ordenaFrequencias[i][1]
				ordenaFrequencias[i][0] = tempValor
				ordenaFrequencias[i][1] = tempIndice

	print("Menores frequências:\n")
	for i in range(5): #Quero as 5 menores frequencias, de acordo com o enunciado
		cincoMenoresFrequencias.append([ordenaFrequencias[i][0], ordenaFrequencias[i][1]])
		print(cincoMenoresFrequencias[i][0])
	print("Índices da menores frequências, na ordem:\n")
	for i in range(5): #Quero as 5 menores frequencias, de acordo com o enunciado
		print(cincoMenoresFrequencias[i][1])	

	return cincoMenoresFrequencias

def devolveModoDeVibracao(modosDeVibracao, indice):
	modoAtual = []
	for i in range(len(modosDeVibracao)):	
		modoAtual.append(modosDeVibracao[i][indice])

	return modoAtual

def main():
	print("1 - Teste da aplicação da transformação de Householder")
	print("2 - Teste da treliça")
	decisao_inicial1 = int(input("Digite o número de teste que gostaria de fazer: "))

	if(decisao_inicial1 == 1): #Teste da aplicação da transformação de Householder
		n = 0 #Inicializando a dimensão n da matriz
		matrizA = [] #Inicializando a matriz A

		print("\n1 - Usar para teste o arquivo 'input-a' ou 'input-b'")
		print("2 - Digitar manualmente os valores de uma matriz real simétrica A")
		decisao_inicial2 = int(input("Digite o número do teste que gostaria de fazer: "))

		if (decisao_inicial2 == 1): #Usar os arquivos prontos 'input-a' ou 'input-b'
			arquivo = ""
			print("\n1 - Usar o arquivo 'input-a'")
			print("2 - Usar o arquivo 'input-b'")
			decisao_inicial3 = int(input("Digite 1 para 'input-a' ou 2 para 'input-b': "))

			if (decisao_inicial3 == 1): #Testes com o arquivo 'input-a'
				arquivo = open("input-a", "r")
			elif (decisao_inicial3 == 2): #Testes com o arquivo 'input-b'
				arquivo = open("input-b", "r")
			
			#APAGARRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR TESTE MEU
			else:
				arquivo = open("input-meu", "r")
			######################################################################################################

			conteudo = arquivo.readlines() #Extraindo todo conteúdo do arquivo escolhido
			arquivo.close() 
			n = int(conteudo[0])
			#print("Número de linhas: ", n)
			for i in range(1, n+1): #Extraindo os valores em si da matriz A
				linha = conteudo[i].split()
				for j in range(n):
					linha[j] = float(linha[j])
				matrizA.append(linha)
			print("\nMatriz A:")
			print(np.array(matrizA)) #Matriz A OK. Falta transformar em ARRAY DE NUMPY

		elif (decisao_inicial2 == 2): #Digitar manualmente os valores de uma matriz real simétrica A
			n = int(input("\nDigite o valor da dimensão n da matriz A simétrica: "))
			print("\nEntre com os valores reais da matriz A")
			for i in range(n):
				print("Linha ", i+1)
				linha = []
				for j in range(n):
					print("Digite o valor da coluna %d: " %(j+1), end="")
					valor = float(input())
					linha.append(valor)
				matrizA.append(linha)
			print("\nMatriz A:")
			print(np.array(matrizA)) #Inicialização da matriz A OK.

		print("\n########################### RESULTADOS ###########################\n")
		T, HT = devolve_TridiagonalSimetrica(matrizA, n)
		print("\nA matriz tridiagonal simétrica é: \n", T)
		print("\nA matriz H transposta é: \n", HT)
		print()

		#Aplicando o Algoritmo QR com deslocamento
		erro = 0.000001
		autovalores, Vk = algoritmoQR_deslocamento(T, HT, n, erro)
		print("\nAutovalores encontrados: \n", autovalores)
		print("\nAutovetores encontrados: \n", Vk)
		print("\n########################### VERIFICAÇÕES PEDIDAS ###########################\n")

		print("Verificando se A * v = lambda * v, para cada autovalor lambda e seu autovetor v correspondente\n")
		for i in range(len(autovalores)):
			print("Verificação para o %dº autovalor:"%(i+1))
			autovetor = devolve_autovetores(Vk, i)
			autovalor = autovalores[i]
			produtoAV = np.array(matrizA) @ autovetor
			produtolambdaV = produtoPorEscalar(autovalor, autovetor)
			print("\nProduto A * V:    ", produtoAV)
			print("Produto lambda * v: ", produtolambdaV, "\n")

		print("\nVerificando se a matriz formada pelos autovetores é ortogonal\n")
		print("Inversa da matriz formada pelos autovetores:\n", np.linalg.inv(Vk))
		print("Transposta da matriz formada pelos autovetores:\n", Vk.T)
			
		
	elif(decisao_inicial1 == 2): #Teste da treliça
		# Extraindo as constantes do arquivo input-c.
		arquivo = open("input-c", "r")
		conteudo = arquivo.readlines() #Extraindo todo conteúdo do arquivo escolhido
		arquivo.close() 
		numNos, nosLivres, numBarras = conteudo[0].split()
		ro, area, elasticidade = conteudo[1].split()
		area = float(area)
		elasticidade = int(elasticidade) * 10**9
		ro = int(ro)
		numNos, nosLivres, numBarras = int(numNos), int(nosLivres), int(numBarras)
		# print(f'numNos: {numNos}')
		# print(f'nosLivres: {nosLivres}')
		# print(f'numBarras: {numBarras}')
		# print(f'ro: {ro}')
		# print(f'area: {area}')
		# print(f'elasticidade: {elasticidade}')
		
		# Extraindo as informações das barras do arquivo input-c.
		counter = 0
		for el in conteudo: counter += 1
		# print(f'counter: {counter}')
		barras = []
		for i in range(2, counter): #Extraindo os valores em si da matriz A ##############################MUDEI AQUI era counter -1
			linha = conteudo[i].split()
			for j in range(4):
				if j < 2:
					linha[j] = int(linha[j])
				else:
					linha[j] = float(linha[j])
			barras.append(linha)
		barras = np.array(barras)
		# print(f'barras:\n{barras}')

		# Procedendo para a definição e montagem das matrizes K e M, em especial.
		K = np.zeros((nosLivres*2,nosLivres*2)) # Matriz de rigidez ##################MUDEI AQUI era 24, tentando generalizar
		M = np.zeros((nosLivres*2,nosLivres*2)) # Matriz de massas  ##################MUDEI AQUI era 24, tentando generalizar

		# Forma as matrizes de rigidez e de massas
		for barra in barras:
			K, M = rotinaDeFormacao(barra, K, M, ro, area, elasticidade)
		#print(f'K (matriz de rigidez total):\n{K}') #SE DESEJAR VER A MATRIZ DE RIGIDEZ TOTAL, DESCOMENTAR ESTA LINHA
		# print(f'M:\n{M}')

		# Faz o calculo de K_til, crucial para a resolução.
		M_inv = np.zeros((nosLivres*2, nosLivres*2)) ##################MUDEI AQUI era 24, tentando generalizar
		for i in range(len(M)):
			for j in range(len(M[0])):
				M_inv[i][j] = math.sqrt(M[i][j])
		M_inv = matrixInverter(M_inv)
		K_til = M_inv @ K @ M_inv
		# print(f'K_til:\n{K_til}')

		# Como K_til é simétrica, podemos triagonalizá-la aplicando Householder.
		T, HT = devolve_TridiagonalSimetrica(K_til, len(K_til))

		# E agora achar os autovalores e autovetores com o algoritmo QR do EP1.
		autovalores, autovetores = algoritmoQR_deslocamento(T, HT, len(K_til), 10**(-15))
		# print(f'autovalores (ref. a frequências):\n{autovalores}')
		# for i in range(len(autovalores)): print(f'{math.sqrt(autovalores[i])}')
		#print("\nOs autovalores encontrados são:\n", autovalores)
		indicesModos = sorted(range(len(autovalores)), key = lambda sub: autovalores[sub])[:5] 

		frequencias = []
		for av in autovalores:
			frequencias.append(math.sqrt(av))

		#print("\nAs frequências de vibração encontradas são:\n", frequencias)
		#print("Os índices dos menores modos de vibração são: \n", indicesModos)
		#print(f'\nModos principais:\n')
		# for i in indicesModos: print(f'{autovetores[i]}')
		# print(f'autovetores:\n{autovetores}')

		# Cálculo dos modos de vibração da treliça.
		y = autovetores
		z = M_inv @ y
		modosDeVibracao = z
		print("\n5 menores frequências e seus respectivos modos de vibração: \n")
		#print(modosDeVibracao)
		#print()
		for coluna in indicesModos:
			#print("Indice: ", coluna)
			print(f'\nFrequências (Hz):')
			print(f'{frequencias[coluna]}')

			print(f'\nModo de vibração:')
			print(devolveModoDeVibracao(modosDeVibracao, coluna))

		print()
		verMatriz = int(input("Se deseja ver a matriz de rididez total, digite 1, se não, digite 0: "))
		if (verMatriz == 1):
			print("\nMatriz de rigidez total:\n")
			print(K)


main()
