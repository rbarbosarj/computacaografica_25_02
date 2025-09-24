# -*- coding: utf-8 -*-
"""
Script completo para a resolução da AC02 de Computação Gráfica.

Este ficheiro implementa as principais transformações geométricas 2D
(translação, escala, rotação, reflexão e cisalhamento) utilizando uma
abordagem profissional com Programação Orientada a Objetos e coordenadas
homogêneas, permitindo a aplicação de transformações através de multiplicação
de matrizes.

O script está dividido em:
1. Funções para criar matrizes de transformação.
2. A classe `FormaGeometrica` que representa os objetos.
3. Funções de plotagem para visualizar os resultados.
4. Funções dedicadas para resolver cada um dos 10 exercícios.
5. A função `main` que orquestra a execução de todos os exercícios.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


# --- Funções para criar Matrizes de Transformação Homogênea (3x3) ---

def matriz_translacao(vetor: Union[List, np.ndarray]) -> np.ndarray:
    """Cria uma matriz de translação 3x3."""
    dx, dy = vetor
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])


def matriz_escala(fatores: Union[float, List, np.ndarray], origem: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Cria uma matriz de escala 3x3 em relação a uma origem."""
    sx, sy = fatores if isinstance(fatores, (list, np.ndarray)) else (fatores, fatores)
    ox, oy = origem
    # Move para a origem, escala, e move de volta
    return matriz_translacao([ox, oy]) @ np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]]) @ matriz_translacao([-ox, -oy])


def matriz_rotacao(angulo_graus: float, origem: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Cria uma matriz de rotação 3x3 em relação a uma origem."""
    angulo_rad = np.radians(angulo_graus)
    c, s = np.cos(angulo_rad), np.sin(angulo_rad)
    ox, oy = origem
    # Move para a origem, rotaciona, e move de volta
    return matriz_translacao([ox, oy]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ matriz_translacao([-ox, -oy])


def matriz_reflexao(eixo: str = 'y') -> np.ndarray:
    """Cria uma matriz de reflexão 3x3."""
    if eixo == 'y': return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if eixo == 'x': return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    raise ValueError("O eixo deve ser 'x' ou 'y'")


def matriz_cisalhamento(k: float, direcao: str = 'horizontal') -> np.ndarray:
    """Cria uma matriz de cisalhamento 3x3."""
    if direcao == 'horizontal': return np.array([[1, k, 0], [0, 1, 0], [0, 0, 1]])
    if direcao == 'vertical': return np.array([[1, 0, 0], [k, 1, 0], [0, 0, 1]])
    raise ValueError("A direção deve ser 'horizontal' ou 'vertical'")


# --- Classe Principal FormaGeometrica ---

class FormaGeometrica:
    def __init__(self, pontos: np.ndarray, nome: str = "Forma"):
        pontos_2d = np.atleast_2d(pontos)
        self.pontos_homogeneos = np.hstack([pontos_2d, np.ones((pontos_2d.shape[0], 1))])
        self.nome = nome

    @property
    def pontos(self) -> np.ndarray:
        return self.pontos_homogeneos[:, :2]

    def __repr__(self) -> str:
        return f"{self.nome}(pontos=\n{self.pontos})"

    def aplicar_matriz(self, matriz: np.ndarray, novo_nome: str) -> 'FormaGeometrica':
        novos_pontos_homogeneos = (matriz @ self.pontos_homogeneos.T).T
        return FormaGeometrica(novos_pontos_homogeneos[:, :2], novo_nome)

    def transladar(self, vetor: Union[List, np.ndarray]) -> 'FormaGeometrica':
        return self.aplicar_matriz(matriz_translacao(vetor), f"{self.nome} transladado")

    def escalar(self, fatores: Union[float, List, np.ndarray],
                origem: Tuple[float, float] = (0, 0)) -> 'FormaGeometrica':
        return self.aplicar_matriz(matriz_escala(fatores, origem), f"{self.nome} escalado")

    def rotacionar(self, angulo_graus: float, origem: Tuple[float, float] = (0, 0)) -> 'FormaGeometrica':
        return self.aplicar_matriz(matriz_rotacao(angulo_graus, origem), f"{self.nome} rotacionado")

    def refletir(self, eixo: str = 'y') -> 'FormaGeometrica':
        return self.aplicar_matriz(matriz_reflexao(eixo), f"{self.nome} refletido")

    def cisalhar(self, k: float, direcao: str = 'horizontal') -> 'FormaGeometrica':
        return self.aplicar_matriz(matriz_cisalhamento(k, direcao), f"{self.nome} cisalhado")


# --- Funções de Plotagem ---

def configurar_plot(ax, titulo: str):
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel("Eixo X")
    ax.set_ylabel("Eixo Y")


def plotar_transformacao(original: FormaGeometrica, transformada: FormaGeometrica, titulo: str):
    fig, ax = plt.subplots(figsize=(8, 8))

    if original.pontos.shape[0] > 1:
        pontos_plot = np.vstack([original.pontos, original.pontos[0]])
        ax.plot(pontos_plot[:, 0], pontos_plot[:, 1], 'o-', label="Original", color='blue')
    else:
        ax.scatter(original.pontos[:, 0], original.pontos[:, 1], s=100, label="Original", color='blue', zorder=5)

    if transformada.pontos.shape[0] > 1:
        pontos_plot = np.vstack([transformada.pontos, transformada.pontos[0]])
        ax.plot(pontos_plot[:, 0], pontos_plot[:, 1], 'o--', label="Transformada", color='red')
    else:
        ax.scatter(transformada.pontos[:, 0], transformada.pontos[:, 1], s=100, label="Transformada", color='red',
                   zorder=5)

    configurar_plot(ax, titulo)
    ax.legend()
    plt.show()


def plotar_transformacao_composta(titulo: str, passos: List[FormaGeometrica]):
    fig, ax = plt.subplots(figsize=(8, 8))
    cores = plt.cm.viridis(np.linspace(0, 1, len(passos)))
    estilos = ['o-', 'o--', 'o-.', 'o:']

    for i, forma in enumerate(passos):
        label = f"Passo {i}: {forma.nome}" if i > 0 else "Original"
        cor = cores[i]
        estilo = estilos[i % len(estilos)]

        if forma.pontos.shape[0] > 1:
            pontos_plot = np.vstack([forma.pontos, forma.pontos[0]])
            ax.plot(pontos_plot[:, 0], pontos_plot[:, 1], estilo, label=label, color=cor, zorder=i + 5)
        else:
            ax.scatter(forma.pontos[:, 0], forma.pontos[:, 1], s=100 + i * 20, label=label, color=cor, zorder=i + 5)

    configurar_plot(ax, titulo)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    plt.show()


# --- Resolução dos Exercícios ---

def resolver_exercicio_1():
    """Resolve e plota o Exercício 1: Translação Simples."""
    print("--- Exercício 1: Translação Simples ---")
    print("Dado o ponto P(2, 3), aplica-se a translação com vetor (4, -2).\n")

    ponto_original = FormaGeometrica(np.array([2, 3]), "P(2, 3)")
    ponto_transformado = ponto_original.transladar([4, -2])

    print(f"Resultado:\nCoordenadas Finais: {ponto_transformado.pontos}\n")
    plotar_transformacao(ponto_original, ponto_transformado, "Exercício 1: Translação Simples")


def resolver_exercicio_2():
    """Resolve e plota o Exercício 2: Escala Uniforme."""
    print("\n--- Exercício 2: Escala Uniforme ---")
    print("Aplica-se uma escala uniforme de fator 2 ao triângulo A(1, 1), B(3, 1), C(2, 4).\n")

    triangulo_original = FormaGeometrica(np.array([[1, 1], [3, 1], [2, 4]]), "Triângulo Original")
    triangulo_transformado = triangulo_original.escalar(2)

    print(f"Resultado:\nNovos vértices:\n{triangulo_transformado.pontos}\n")
    plotar_transformacao(triangulo_original, triangulo_transformado, "Exercício 2: Escala Uniforme")


def resolver_exercicio_3():
    """Resolve e plota o Exercício 3: Escala Não Uniforme."""
    print("\n--- Exercício 3: Escala Não Uniforme ---")
    print("Aplica-se uma escala com fatores (x=2, y=0.5) ao mesmo triângulo.\n")

    triangulo_original = FormaGeometrica(np.array([[1, 1], [3, 1], [2, 4]]), "Triângulo Original")
    triangulo_transformado = triangulo_original.escalar([2, 0.5])

    print(f"Resultado:\nNovos vértices:\n{triangulo_transformado.pontos}\n")
    plotar_transformacao(triangulo_original, triangulo_transformado, "Exercício 3: Escala Não Uniforme")


def resolver_exercicio_4():
    """Resolve e plota o Exercício 4: Rotação na Origem."""
    print("\n--- Exercício 4: Rotação na Origem ---")
    print("Rotaciona-se o ponto P(1, 0) em 90° no sentido anti-horário.\n")

    ponto_original = FormaGeometrica(np.array([1, 0]), "P(1, 0)")
    ponto_transformado = ponto_original.rotacionar(90)

    print(f"Resultado:\nCoordenadas Finais: {np.round(ponto_transformado.pontos, 5)}\n")
    plotar_transformacao(ponto_original, ponto_transformado, "Exercício 4: Rotação 90° Anti-horário")


def resolver_exercicio_5():
    """Resolve e plota o Exercício 5: Rotação de um Polígono."""
    print("\n--- Exercício 5: Rotação de um Polígono ---")
    print("Rotaciona-se um quadrado em 45° no sentido horário.\n")

    quadrado_original = FormaGeometrica(np.array([[1, 1], [1, 4], [4, 4], [4, 1]]), "Quadrado Original")
    quadrado_transformado = quadrado_original.rotacionar(-45)  # Ângulo negativo para sentido horário

    print(f"Resultado:\nNovos vértices:\n{np.round(quadrado_transformado.pontos, 5)}\n")
    plotar_transformacao(quadrado_original, quadrado_transformado, "Exercício 5: Rotação 45° Horário")


def resolver_exercicio_6():
    """Resolve e plota o Exercício 6: Reflexão Simples."""
    print("\n--- Exercício 6: Reflexão Simples ---")
    print("Reflete-se o ponto P(2, 5) em relação ao eixo Y.\n")

    ponto_original = FormaGeometrica(np.array([2, 5]), "P(2, 5)")
    ponto_transformado = ponto_original.refletir(eixo='y')

    print(f"Resultado:\nCoordenadas Finais: {ponto_transformado.pontos}\n")
    plotar_transformacao(ponto_original, ponto_transformado, "Exercício 6: Reflexão no Eixo Y")


def resolver_exercicio_7():
    """Resolve e plota o Exercício 7: Reflexão de um Triângulo."""
    print("\n--- Exercício 7: Reflexão de um Triângulo ---")
    print("Reflete-se o triângulo A(2,3), B(4,3), C(3,5) em relação ao eixo X.\n")

    triangulo_original = FormaGeometrica(np.array([[2, 3], [4, 3], [3, 5]]), "Triângulo Original")
    triangulo_transformado = triangulo_original.refletir(eixo='x')

    print(f"Resultado:\nNovos vértices:\n{triangulo_transformado.pontos}\n")
    plotar_transformacao(triangulo_original, triangulo_transformado, "Exercício 7: Reflexão no Eixo X")


def resolver_exercicio_8():
    """Resolve e plota o Exercício 8: Cisalhamento Horizontal."""
    print("\n--- Exercício 8: Cisalhamento Horizontal ---")
    print("Aplica-se um cisalhamento horizontal com k=2 ao ponto P(2, 3).\n")

    ponto_original = FormaGeometrica(np.array([2, 3]), "P(2, 3)")
    ponto_transformado = ponto_original.cisalhar(k=2, direcao='horizontal')

    print(f"Resultado:\nCoordenadas Finais: {ponto_transformado.pontos}\n")
    plotar_transformacao(ponto_original, ponto_transformado, "Exercício 8: Cisalhamento Horizontal (k=2)")


def resolver_exercicio_9():
    """Resolve e plota o Exercício 9: Composição de Transformações."""
    print("\n--- Exercício 9: Composição de Transformações ---")
    print("Aplica-se ao ponto P(3, 2) a sequência:\n"
          "1. Translação com vetor (1, -1)\n"
          "2. Rotação de 90° no sentido anti-horário\n"
          "3. Escala uniforme com fator 2\n")

    ponto_original = FormaGeometrica(np.array([3, 2]), "Original")
    passo_1 = ponto_original.transladar([1, -1])
    passo_2 = passo_1.rotacionar(90)
    passo_3_final = passo_2.escalar(2)

    print(f"Resultado:\nPasso 1 (Translação): {passo_1.pontos}"
          f"\nPasso 2 (Rotação): {np.round(passo_2.pontos, 5)}"
          f"\nPasso 3 (Final): {np.round(passo_3_final.pontos, 5)}\n")

    plotar_transformacao_composta(
        "Exercício 9: Composição de Transformações",
        [ponto_original, passo_1, passo_2, passo_3_final]
    )


def resolver_exercicio_10():
    """Resolve e plota o Exercício 10: Combinação de Transformações."""
    print("\n--- Exercício 10: Combinação de Transformações ---")
    print("Aplica-se ao retângulo A(1,1), B(5,1), C(5,3), D(1,3) a sequência:\n"
          "1. Translação com vetor (-2, 3)\n"
          "2. Escala não uniforme com fatores (1.5, 0.5)\n"
          "3. Reflexão em relação ao eixo Y\n")

    retangulo_original = FormaGeometrica(np.array([[1, 1], [5, 1], [5, 3], [1, 3]]), "Original")

    # Método 1: Calculando a matriz de transformação final (mais eficiente)
    matriz_final = (
            matriz_reflexao('y') @
            matriz_escala([1.5, 0.5]) @
            matriz_translacao([-2, 3])
    )
    retangulo_final = retangulo_original.aplicar_matriz(matriz_final, "Final")
    print(f"Resultado (com matriz composta):\nNovos vértices:\n{np.round(retangulo_final.pontos, 5)}\n")

    # Método 2: Aplicando passo a passo para visualização
    passo_1 = retangulo_original.transladar([-2, 3])
    passo_2 = passo_1.escalar([1.5, 0.5])
    passo_3_final = passo_2.refletir(eixo='y')

    plotar_transformacao_composta(
        "Exercício 10: Combinação de Transformações",
        [retangulo_original, passo_1, passo_2, passo_3_final]
    )


# --- Função Principal para Executar os Exercícios ---

def main():
    """Função principal que orquestra a resolução de todos os exercícios."""
    print("=" * 60)
    print("Executando AC02 - Transformações Geométricas")
    print("=" * 60)

    resolver_exercicio_1()
    resolver_exercicio_2()
    resolver_exercicio_3()
    resolver_exercicio_4()
    resolver_exercicio_5()
    resolver_exercicio_6()
    resolver_exercicio_7()
    resolver_exercicio_8()
    resolver_exercicio_9()
    resolver_exercicio_10()

    print("\n" + "=" * 60)
    print("Todos os exercícios foram executados.")
    print("Pode fechar as janelas dos gráficos para terminar o programa.")
    print("=" * 60)


# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    main()

