# -*- coding: utf-8 -*-
"""
# AVD: Trabalho 1 - ADbib

Ágata Clícia Santos Brazão - 22152248

Ricardo Nogueira Miranda Filho - 22153740
"""

from scipy import stats # biblioteca para uso do T Student bicaudal
import matplotlib.pyplot as plt # biblioteca apenas para a plotagem dos gráficos
from math import * # para usar a biblioteca de raiz quadrada, equivalente a elevar a 1/2
import random
import numpy as np

"""## Funções Auxiliares"""

def coeficiente_variacao_aux(amostra,media):
    # Calcula o desvio padrão

    tam_amostra = len(amostra)

    variancia = (1 / (tam_amostra - 1)) * sum([(elem - media)**2 for elem in amostra])

    desvio = sqrt(variancia)

    # Calcula o coeficiente de variação
    coeficiente_variacao = desvio / media

    resultados = {
        'coef': coeficiente_variacao,
        'decisao': (coeficiente_variacao < 0.5)
    }

    return resultados

"""## Implementações"""

# Função para encontrar a distribuição de frequência para uma dada amostra. Código de Identificação: F01
# F01
def dfreq(amostra):
    amostra.sort()
    frequencia = {}

    for elem in set(amostra):
        frequencia[elem] = 1

    for index, elem in enumerate(amostra):
        if(index > 0 and elem == amostra[index - 1]):
            frequencia[elem] = frequencia[elem] + 1

    return frequencia

# A função recebe uma lista como parâmetro, a lista é ordenada. Primeiro um for itera sobre um set da amostra
# então não existe elementos repetidos, e coloca o elemento como uma chave de um dicionário, e coloca o valor 1
# o segundo for itera sobre a lista, verificando se o elemento atual é igual ao anterior na lista, caso seja
# ele incrementa 1 na chave correspondente ao elemento, sendo assim ao fim tem-se um dicionário com frequência
# dos elementos, sendo a chave o elemento e a frequência o valor indicado pela chave.

# Função para plotar o histograma de frequência, recebe como parâmetro o dicionário da função anterior.
def plot_histograma(frequencia):
    # Preparando os dados
    valores = list(frequencia.keys())
    frequencias = list(frequencia.values())

    # Criando o histograma
    plt.figure(figsize=(10, 5))
    plt.bar(valores, frequencias, color='blue')

    # Adicionando título e labels
    plt.title('Histograma de Frequência')
    plt.xlabel('Elementos')
    plt.ylabel('Frequência')

    # Mostrando o gráfico
    plt.show()

# Função que, dada uma amostra, analise os critérios para selecionar o cálculo mais apropriado para encontrar a média. Código de Identificação: F02
# F02
# Média Aritmética, Ponderada , Geométrica, Harmônica, de Taxas, Moda e Mediana

def mediaAritmetica(amostra):
    return sum(amostra) / len(amostra)

def mediana(amostra):
    tam = len(amostra)
    meio = tam // 2

    if tam % 2 == 0: mediana = (amostra[meio - 1] + amostra[meio]) / 2
    else: mediana = amostra[meio]
    return mediana

def escolhaMedia(amostra, pesos=None):

    # cabeçalho
    amostra.sort()
    tamanho = len(amostra)
    frequencias = {}
    meio = tamanho // 2
    produto = 1
    for n in amostra: produto *= n
    temNum = 0

    aritmetica, ponderada, geometrica, harmonica, mediana, moda = 0,0,0,0,0,0
    resultados = {
        'melhorMedia': 0,
        'aritmetica': 0,
        'ponderada': 0,
        'geometrica': 0,
        'harmonica': 0,
        'mediana': 0,
        'moda': 0
    }

    # cálculos
    if not any(isinstance(elemento, str) for elemento in amostra):
        temNum = 1

        aritmetica = sum(amostra) / tamanho
        resultados['aritmetica'] = aritmetica

        try: ponderada = sum(n * p for n, p in zip(amostra, pesos)) / sum(pesos)
        except: ponderada = aritmetica
        resultados['ponderada'] = ponderada

        geometrica = produto**(1/tamanho)
        resultados['geometrica'] = geometrica

        try: harmonica = tamanho / sum(1/n for n in amostra)
        except: harmonica = aritmetica
        resultados['harmonica'] = harmonica
        # taxas = sum(t * q for t, q in zip(taxas, quantidades)) / sum(quantidades)

        if n % 2 == 0: mediana = (amostra[meio - 1] + amostra[meio]) / 2
        else: mediana = amostra[meio]
        resultados['mediana'] = mediana

    for elemento in amostra:
        if elemento in frequencias: frequencias[elemento] += 1
        else: frequencias[elemento] = 1
    maiorFrequencia = max(frequencias.values())
    modas = [chave for chave, valor in frequencias.items() if valor == maiorFrequencia]
    resultados['moda'] = modas

    coef_median = coeficiente_variacao_aux(amostra, mediana)
    coef_arit = coeficiente_variacao_aux(amostra, aritmetica)
    coef_geo = 0
    if all(x >= 0 for x in amostra): coef_geo = coeficiente_variacao_aux(amostra, geometrica)

    #decisão
    if not (temNum): media = modas
    elif pesos: media = ponderada
    elif ((amostra[0]/amostra[-1]) <= 0.5) and coef_median['decisao']: media = mediana
    elif (all(x >= 0 for x in amostra)) and coef_geo['decisao'] and coef_geo['coef'] < coef_arit['coef']: media = geometrica
    elif coef_arit['decisao']: media = aritmetica
    else: media = harmonica

    resultados['melhorMedia'] = media
    return resultados

# Explicação das decisões:
# Moda: Se o dado for categórico, a melhor média já é definida como a moda;
# Média Ponderada: Se for indicado pesos, já é definida como a melhor média;
# A mediana é definida como melhor média se a razao do menor valor com o maior valor for menor que 50% e
# o coeficiente de variação for menor que 50%. Isso porque a mediana é boa para amostras com outliers e o
# coef serve para verificar se está representativo;
# A geométrica é definida como melhor se a amostra for inteiramente positiva e a mesma questão do coef;
# A aritmetica quisemos deixar como a mais geral, basta o coef dar True na função que ela pode ser escolhida;
# A harmonica não foi tao trabalhada, pois vimos que ela é a menos utilizada, então se mais nenhuma passar a
# ideia é ela ser a escolhida.

# Função que, dada uma amostra, encontre as medidas de dispersão estudadas. Código de Identificação: F03
# F03
# Amplitude, Variância Amostral, Desvio Padrão, Coeficiente de Variação, Quartis, Amplitude Interquartil
# Para evitar a poluição do notebook, o gráfico de boxplot é opcional, basta passar como True o parâmetro graph.

def desvioPadrao(amostra):
    tam_amostra = len(amostra)
    media = mediaAritmetica(amostra)
    variancia = (1 / (tam_amostra - 1)) * sum([(elem - media)**2 for elem in amostra])
    return sqrt(variancia)

def medDisp(amostra, boxplot=False, dispersao=False):
    amostra.sort()

    tam_amostra = len(amostra)
    meio = tam_amostra // 2
    media = mediaAritmetica(amostra)
    if tam_amostra % 2 == 0: mediana = (amostra[meio - 1] + amostra[meio]) / 2
    else: mediana = amostra[meio]

    amplitude = amostra[-1] - amostra[0]
    variancia = (1 / (tam_amostra - 1)) * sum([(elem - media)**2 for elem in amostra])
    desvio = sqrt(variancia)
    coeficiente = (desvio / media) * 100

    q1 = amostra[int(0.25 * (tam_amostra + 1)) - 1]
    q2 = mediana
    q3 = amostra[int(0.75 * (tam_amostra + 1)) - 1]
    aiq = q3 - q1  # Amplitude Interquartil

    quartis = [q1, q2, q3]

    medidas = {}

    medidas["amplitude"] = amplitude
    medidas["variancia"] = variancia
    medidas["desvio"] = desvio
    medidas["coeficiente"] = coeficiente
    medidas["quartis"] = quartis
    medidas["amp_qu"] = aiq

    if(boxplot):
        plt.boxplot(amostra, vert=False, patch_artist=True)
        plt.title("Boxplot dos Quartis")
        plt.xlabel("Valores da Amostra")
        plt.show()

    if(dispersao):
        plt.scatter(range(len(amostra)), amostra, color='blue')
        plt.title("Gráfico de Dispersão da Amostra")
        plt.xlabel("Índice")
        plt.ylabel("Valores")
        plt.show()

    return medidas

# Nessa função foi decidido usar apenas a média aritmética, pois para os cálculos é a mais usada.

# Função que, dada uma amostra, encontre o Intervalo de Confiança (IC). Código de Identificação: F04
# F04

def intervaloConfianca(amostra, media=None, pesos=None, nivel=0.95):
    tamanhoAmostra = len(amostra)
    if not media: mediaAmostral = mediaAritmetica(amostra)
    else: mediaAmostral = media
    variancia = (1 / (len(amostra) - 1)) * sum([(elem - mediaAmostral)**2 for elem in amostra])
    desvio = math.sqrt(variancia)

    erroPadrao = desvio / tamanhoAmostra**(1/2)
    grausLib = tamanhoAmostra - 1

    return stats.t.interval(nivel, df=grausLib, loc=mediaAmostral, scale=erroPadrao)

# Diferente da função medDisp(), nessa função foi utilizada a media retornada da escolhaMedia(). Porém, por
# padrão ela usa a aritmética também, podendo ser mudada a média por parâmetro.

# Função que, escolhido um nível de confiança, possa identificar se duas amostras são significativamente diferentes. Código de Identificação: F05
# F05

def comparar_amostras(amostraA, amostraB, nivel_confianca):

    media_A, desvio_A = escolhaMedia(amostraA)['melhorMedia'], medDisp(amostraA)['desvio']
    media_B, desvio_B = escolhaMedia(amostraB)['melhorMedia'], medDisp(amostraB)['desvio']

    t_value, p_value = stats.ttest_ind(amostraA, amostraB, equal_var=False)

    if p_value < 1 - nivel_confianca:
        return True
    return False

# Função que, baseada em uma amostra preliminar de
# um experimento, possa indicar o tamanho mínimo da amostra a ser
# coleta para alcançar um determinado nível de confiança nos resultados. Código de Identificação: F06
# F06

def tamAmostra(desvio_padrao, media = 1, confianca=95, precisao_desejada=3):
    # Converter o nível de confiança para uma escala de 0 a 1
    confianca /= 100

    # Encontrar o valor de Z correspondente ao nível de confiança desejado
    # norm.ppf retorna o valor de Z para a cauda à esquerda, então usamos (1 + confianca)/2 para o centro
    Z = stats.norm.ppf((1 + confianca) / 2)

    # A precisão desejada é metade da largura do intervalo de confiança
    E = precisao_desejada / 2

    # Calculando o tamanho da amostra
    n = ((100 * Z * desvio_padrao) / (E*media)) ** 2
    return ceil(n)/100  # Arredonda para cima para garantir a precisão

# Implementar uma função que, baseada em uma amostra preliminar de
# um experimento, possa indicar o tamanho mínimo da amostra a ser
# coleta para alcançar um determinado nível de confiança nos resultados.

def calcular_tamanho_amostra(amostra, confianca):

    media = escolhaMedia(amostra)['melhorMedia']

    desvio = medDisp(amostra)['desvio']

    valores_unicos = sorted(set(amostra))
    precisao_desejada = min(abs(valores_unicos[i] - valores_unicos[i-1]) for i in range(1, len(valores_unicos)))

    if precisao_desejada == 0:
        precisao_desejada = 0.1

    confianca /= 100

    Z = stats.norm.ppf((1 + confianca) / 2)

    n = (Z * desvio / precisao_desejada) ** 2
    return ceil(n)/100  # Arredonda para cima para garantir a precisão

def aleatoriaQualquer(valores, probabilidades):

    e_x = sum([a * b for a, b in zip(valores, probabilidades)])
    e_x2 = sum([pow(a, 2) * b for a, b in zip(valores, probabilidades)])
    var_x = e_x2 - pow(e_x, 2)

    acumulado = 0
    f_x = []
    valores_ordenados = sorted(set(valores))
    for valor in valores_ordenados:
        acumulado += sum(prob for val, prob in zip(valores, probabilidades) if val == valor)
        f_x.append(acumulado)

    plt.step(valores_ordenados, f_x, where="post")
    plt.xlabel('Valores')
    plt.ylabel('F(x)')
    plt.title('Função de Distribuição Acumulada (FDA)')
    plt.grid(True)
    plt.show()

    return {
        "media": e_x,
        "variancia": var_x
    }

"""Variáveis Aleatórias Discretas"""

def bernoulli(p):

    valor_esperado = p

    variancia = p * (1 - p)

    if p != 0:
        coeficiente_variacao = sqrt(variancia) / valor_esperado
    else:
        coeficiente_variacao = float('inf')

    return valor_esperado, variancia, coeficiente_variacao

def binomial(n, p):
    valor_esperado = n * p

    variancia = n * p * (1 - p)

    if valor_esperado != 0:
        coeficiente_variacao = sqrt(variancia) / valor_esperado
    else:
        coeficiente_variacao = float('inf')

    return valor_esperado, variancia, coeficiente_variacao

def probabilideBinomial(n, k, p):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def geometrica(p):
    valor_esperado = 1 / p  # Valor esperado
    variancia = (1 - p) / (p**2)  # Variância
    coeficiente_variacao = ((1 - p) / p**2)**0.5 / (1 / p)  # Coeficiente de variação

    return valor_esperado, variancia, coeficiente_variacao

def probabilidade_geométrica(p, i):
    return p * ((1 - p) ** (i - 1))

def poisson(lambd):
    valor_esperado = lambd

    variancia = lambd

    coeficiente_variacao = sqrt(lambd) / lambd

    return valor_esperado, variancia, coeficiente_variacao

def probabilidade_poisson(lambd, i):
    return (exp(-lambd) * (lambd ** i)) / factorial(i)

"""Variáveis Aleatórias Contínuas"""

def varUniforme(a, b):
    valor_esperado = (a + b) / 2

    variancia = ((b - a) ** 2) / 12

    coeficiente_variacao = (sqrt(variancia) / valor_esperado) * 100

    return valor_esperado, variancia, coeficiente_variacao

def varExponencial(a):
    valor_esperado = a
    variancia = a**2
    coeficiente_variacao = 1

    return valor_esperado, variancia, coeficiente_variacao

def varNormal(media, variancia):
    valor_esperado = media  # Valor esperado
    var_X = variancia  # Variância
    coeficiente_variacao = (variancia**0.5) / media  # Coeficiente de variação

    return valor_esperado, var_X, coeficiente_variacao

def variavel_aleatoria_exponencial(beta):
    U = random.random()
    X = -beta * log(1 - U)
    return X

def regra_chow_robbins(dados, d, nivel_confianca=0.95):
    
    n = len(dados)
    if n < 2:
        return False, (0,0)
    
    media = mediaAritmetica(dados)
    intervalo_confianca = intervaloConfianca(dados, media, nivel=nivel_confianca)
    limite_inferior, limite_superior = intervalo_confianca
    
    largura_ic = limite_superior - limite_inferior
    deve_parar = largura_ic < d
    
    return deve_parar, intervalo_confianca

def regra_intervalo_confianca_relativo(data, gamma):
    media = np.mean(data)
    ic = intervaloConfianca(data, media)
    h = (ic[1] - ic[0]) / 2
    precision = h / media
    return precision <= gamma, ic, h


def conaway(amostra):
    menor = float("inf")
    maior = float("-inf")

    for i, elem in enumerate(amostra):
        if elem < menor:
            menor = elem
        if elem > maior:
            maior = elem

        transiente = False
        for j in range(i + 1, len(amostra)):
            if amostra[j] <= menor or amostra[j] >= maior:
                transiente = True
                break

        if not transiente:
            return i

    return -1

def fishman(amostra, k):
    media = mediaAritmetica(amostra)
    cruzou = 0

    for i, elem in enumerate(amostra):
        if i > 0 and elem > media and amostra[i - 1] < media: cruzou += 1
        elif i > 0 and elem < media and amostra[i - 1] > media: cruzou += 1
        
        if cruzou == k: return len(amostra) - i, i

    return 0, 0

def mm1_v3(_lambda, _myu, n, d=None, gamma=None, conway=False, fish=None, pr = 0.05, filename="tempos_de_espera.txt", file=False):
    # Inicialização de variáveis usando listas
    chegadas = [variavel_aleatoria_exponencial(1 / _lambda) for _ in range(n)]
    servicos = [variavel_aleatoria_exponencial(1 / _myu) for _ in range(n)]
    
    horaChegada = [sum(chegadas[:i+1]) for i in range(n)]
    inicioServico = [0] * n
    fimServico = [0] * n
    espera = [0] * n
    
    # Primeira chegada e término de serviço
    inicioServico[0] = horaChegada[0]
    fimServico[0] = inicioServico[0] + servicos[0]
    espera[0] = inicioServico[0] - horaChegada[0]
    
    # Simulação da fila M/M/1
    for i in range(1, n):
        inicioServico[i] = max(horaChegada[i], fimServico[i - 1])
        fimServico[i] = inicioServico[i] + servicos[i]
        espera[i] = inicioServico[i] - horaChegada[i]

    # Processamento adicional caso d ou gamma sejam fornecidos
    clientes = n
    while True:
        if d is not None:
            parada, ic = regra_chow_robbins(espera[:clientes], d)
            if parada:
                break
        elif gamma is not None:
            parada, ic, h = regra_intervalo_confianca_relativo(espera[:clientes], gamma)
            if parada:
                break
        elif conway:
            pass
        elif fish is not None:
            pass
        else:
            break
            
        
        # Adicionar mais 'n' clientes
        if fish:
            fishcar, fish_i = fishman(espera, fish)
            n = n - fishcar

        if conway:
            conway_i = conaway(espera)
            conwaycar = len(espera) - conway_i
            n = n - conwaycar

        novas_chegadas = [variavel_aleatoria_exponencial(1 / _lambda) for _ in range(n)]
        novos_servicos = [variavel_aleatoria_exponencial(1 / _myu) for _ in range(n)]
        
        novas_horasChegada = [horaChegada[-1] + sum(novas_chegadas[:i+1]) for i in range(n)]
        horaChegada.extend(novas_horasChegada)
        inicioServico_novos = [0] * n
        fimServico_novos = [0] * n
        espera_novos = [0] * n

        for i in range(n):
            if i == 0:
                inicioServico_novos[i] = max(novas_horasChegada[i], fimServico[-1])
            else:
                inicioServico_novos[i] = max(novas_horasChegada[i], fimServico_novos[i - 1])
            fimServico_novos[i] = inicioServico_novos[i] + novos_servicos[i]
            espera_novos[i] = inicioServico_novos[i] - novas_horasChegada[i]
        
        inicioServico.extend(inicioServico_novos)
        fimServico.extend(fimServico_novos)
        espera.extend(espera_novos)
        
        clientes += n

        if fish is not None and fishcar != 0:
            espera = espera[fish_i:fish_i + n + fishcar]
            clientes = len(espera)
            break

        if conway and conway_i != -1:
            espera = espera[conway_i:conway_i + n + conwaycar]
            clientes = len(espera)
            break

    if file:
        # Escrever os tempos de espera em um arquivo de texto
        with open(filename, 'w') as f:
            for e in espera[:clientes]:
                f.write(f"{e}\n")
    
    
    # Cálculo do tempo médio de espera
    media = sum(espera[:clientes]) / clientes
    confianca = intervaloConfianca(espera[:clientes], media)

    # Valor Teórico
    rho = _lambda / _myu
    esperado = (rho / (_myu * (1 - rho)))

    # Resultados
    resultados = {
        'medio': media,
        'confianca': (confianca[0], confianca[1]),
        'esperado': esperado,
        'duracao': fimServico[clientes-1] - inicioServico[0],
        'n': clientes,
        'h': (confianca[1] - confianca[0]) / 2
    }

    if gamma is not None: resultados['h'] = h

    return resultados

def mm1_v4(_lambda, _myu, n, d=None, gamma=None, filename="tempos_de_espera.txt", file=False):
    # Inicialização de variáveis usando Numpy
    chegadas = np.array([variavel_aleatoria_exponencial(1 / _lambda) for _ in range(n)])
    servicos = np.array([variavel_aleatoria_exponencial(1 / _myu) for _ in range(n)])
    
    horaChegada = np.cumsum(chegadas)
    inicioServico = np.zeros(n)
    fimServico = np.zeros(n)
    espera = np.zeros(n)
    
    # Primeira chegada e término de serviço
    inicioServico[0] = horaChegada[0]
    fimServico[0] = inicioServico[0] + servicos[0]
    espera[0] = inicioServico[0] - horaChegada[0]
    
    # Simulação da fila M/M/1
    for i in range(1, n):
        inicioServico[i] = max(horaChegada[i], fimServico[i - 1])
        fimServico[i] = inicioServico[i] + servicos[i]
        espera[i] = inicioServico[i] - horaChegada[i]

    # Processamento adicional caso d ou gamma sejam fornecidos
    clientes = n
    while True:
        if d is not None:
            parada, ic = regra_chow_robbins(espera[:clientes], d)
            if parada:
                break
        elif gamma is not None:
            parada, ic, h = regra_intervalo_confianca_relativo(espera[:clientes], gamma)
            if parada:
                break
        else:
            break
        
        # Adicionar mais 'n' clientes
        novas_chegadas = np.array([variavel_aleatoria_exponencial(1 / _lambda) for _ in range(n)])
        novos_servicos = np.array([variavel_aleatoria_exponencial(1 / _myu) for _ in range(n)])
        
        novas_horasChegada = np.cumsum(novas_chegadas) + horaChegada[-1]
        horaChegada = np.append(horaChegada, novas_horasChegada)
        inicioServico_novos = np.zeros(n)
        fimServico_novos = np.zeros(n)
        espera_novos = np.zeros(n)

        for i in range(n):
            if i == 0:
                inicioServico_novos[i] = max(novas_horasChegada[i], fimServico[-1])
            else:
                inicioServico_novos[i] = max(novas_horasChegada[i], fimServico_novos[i - 1])
            fimServico_novos[i] = inicioServico_novos[i] + novos_servicos[i]
            espera_novos[i] = inicioServico_novos[i] - novas_horasChegada[i]
        
        inicioServico = np.append(inicioServico, inicioServico_novos)
        fimServico = np.append(fimServico, fimServico_novos)
        espera = np.append(espera, espera_novos)
        
        clientes += n

    if file:
        # Escrever os tempos de espera em um arquivo de texto
        np.savetxt(filename, espera[:clientes], fmt='%f')
    
    # Cálculo do tempo médio de espera
    media = np.mean(espera[:clientes])
    confianca = intervaloConfianca(espera[:clientes], media)

    # Valor Teórico
    rho = _lambda / _myu
    esperado = (rho / (_myu * (1 - rho)))

    # Resultados
    resultados = {
        'medio': media,
        'confianca': (confianca[0], confianca[1]),
        'esperado': esperado,
        'duracao': fimServico[clientes-1] - inicioServico[0],
        'n': clientes,
        'h': (confianca[1] - confianca[0]) / 2
    }

    if gamma is not None: resultados['h'] = h

    return resultados


# def MSER_5Y(espera):
#     n = len(espera)
#     min_mse = float('inf')
#     best_index = 0
#     for i in range(n // 10, n // 2):
#         subset = espera[i:]
#         mean = np.mean(subset)
#         mse = np.mean((subset - mean) ** 2)
#         if mse < min_mse:
#             min_mse = mse
#             best_index = i
#     print(best_index)
#     return best_index

def MSER_5Y(espera, K):
    n = len(espera)
    m = n // K
    min_mse = float('inf')
    ponto_truncagem = 0
    for i in range(m, n // 2, m):
        conjunto = espera[i:]
        media = mediaAritmetica(conjunto)
        mse = mediaAritmetica((conjunto - media) ** 2)
        if mse < min_mse:
            min_mse = mse
            ponto_truncagem = i
    # print(ponto_truncagem)
    return ponto_truncagem

def funcao_von_neumann(espera, K, beta):
    n = len(espera)
    m = n // K
    W = np.array([mediaAritmetica(espera[j*m:(j+1)*m]) for j in range(K)])
    W_bar = mediaAritmetica(W)
    S2_W = np.sum((W - W_bar)**2) / (K - 1)
    MSSD_W = np.sum((W[1:] - W[:-1])**2) / (K - 1)
    C_v = 1 - (MSSD_W / (2 * S2_W))
    # criterio de teste
    test_statistic = C_v / np.sqrt((K - 2) / ((K**2) - 1))
    
    # valor critico
    z_beta = stats.norm.ppf(1 - beta)
    
    # teste randomico
    passou_no_teste = test_statistic <= z_beta
    
    return passou_no_teste

def mm1_vMSER5Y(_lambda, _myu, n, gamma=0.05, K=10):
    # Inicialização de variáveis usando Numpy
    chegadas = np.array([variavel_aleatoria_exponencial(1 / _lambda) for _ in range(n)])
    servicos = np.array([variavel_aleatoria_exponencial(1 / _myu) for _ in range(n)])
    
    horaChegada = np.cumsum(chegadas)
    inicioServico = np.zeros(n)
    fimServico = np.zeros(n)
    espera = np.zeros(n)
    
    # Primeira chegada e término de serviço
    inicioServico[0] = horaChegada[0]
    fimServico[0] = inicioServico[0] + servicos[0]
    espera[0] = inicioServico[0] - horaChegada[0]
    
    # Simulação da fila M/M/1
    for i in range(1, n):
        inicioServico[i] = max(horaChegada[i], fimServico[i - 1])
        fimServico[i] = inicioServico[i] + servicos[i]
        espera[i] = inicioServico[i] - horaChegada[i]

    # Eliminação do transiente usando MSER-5Y com coleta adicional de dados
    while True:
        corte = MSER_5Y(espera,K)
        # espera = espera[corte:]
        
        if funcao_von_neumann(espera, K, gamma):
            # Aplicação da regra de parada com precisão relativa de 5%
            clientes = len(espera)
            parada, ic, h = regra_intervalo_confianca_relativo(espera, gamma)
            if parada:
                # print("antes corte", len(espera))
                espera = espera[corte:]
                # print("depois corte", len(espera))
                break

        # Adicionar mais 'n' clientes
        novas_chegadas = np.array([variavel_aleatoria_exponencial(1/_lambda) for _ in range(n)])
        novos_servicos = np.array([variavel_aleatoria_exponencial(1/_myu) for _ in range(n)])
        
        novas_horasChegada = np.cumsum(novas_chegadas) + horaChegada[-1]
        horaChegada = np.append(horaChegada, novas_horasChegada)
        inicioServico_novos = np.zeros(n)
        fimServico_novos = np.zeros(n)
        espera_novos = np.zeros(n)

        for i in range(n):
            if i == 0:
                inicioServico_novos[i] = max(novas_horasChegada[i], fimServico[-1])
            else:
                inicioServico_novos[i] = max(novas_horasChegada[i], fimServico_novos[i - 1])
            fimServico_novos[i] = inicioServico_novos[i] + novos_servicos[i]
            espera_novos[i] = inicioServico_novos[i] - novas_horasChegada[i]
        
        inicioServico = np.append(inicioServico, inicioServico_novos)
        fimServico = np.append(fimServico, fimServico_novos)
        espera = np.append(espera, espera_novos)

    # Cálculo do tempo médio de espera
    media = mediaAritmetica(espera)
    confianca = intervaloConfianca(espera, media)

    # Valor Teórico
    rho = _lambda / _myu
    esperado = (rho / (_myu * (1 - rho)))

    # Resultados
    resultados = {
        'medio': media,
        'confianca': (confianca[0], confianca[1]),
        'esperado': esperado,
        'duracao': fimServico[clientes-1] - inicioServico[0],
        'n': clientes,
        'h': h
    }

    return resultados
