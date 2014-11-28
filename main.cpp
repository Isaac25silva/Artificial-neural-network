#include <iostream>
#include <stdio.h>
#include "classe.h"
#include <fstream>
#include <vector>

using namespace std;

int main()
{
	//double m[7000][1000];

	//-- VariÃ¡veis utilizada somente em treinamentos novos--
	double n = 0.45;
	double alpha = 0.2;
	unsigned int  numEntradas = 256; //Entre com o numero de entradas da RNA
	unsigned int numSaidas = 10;	   //Entre com o numero de saidas da RNA
	double Wo = 0; //bias
	int numCamadaEscondida = 4;
    	unsigned int iteracoes = 100;    //Iterações usadas apenas no treinamento novo
	int numNeuronios = 300;
	//----------------------------------------------

    std::string train = "zip.train.txt"; //Nome do arquivo para carregar a tabela
    std::string test = "zip.test.txt"; //Nome do arquivo para carregar a tabela
    std::string learned = "learned120.dat"; //Nome do arquivo para salvar ou carregar treinamento

    //std::vector<std::string> entrada1, entrada2;

     RNA Rna;//Cria o objeto

     //Rna.Rna(train, 257); // realiza o treinamento

     //Rna.runNewTraining(train, learned, Wo, numNeuronios, alpha, n, numCamadaEscondida, numEntradas, iteracoes, numSaidas);

     //Rna.runTraining(train, learned); // realiza o treinamento a partir de um arquivo de treinamento anterior

     Rna.runLearnedFile(test, learned); // realiza o teste na rede treinada

  return 0;
}




