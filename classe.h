/*-------------------------------------------------------------------------
***************************************************************************
* @file classe.h
* @author Isaac Jesus da Silva - FEI
* @version V0.0.1
* @created 30/10/2014
* @Modified 04/11/2014
* @e-mail isaac25silva@yahoo.com.br
***************************************************************************
Arquivo de cabeçalho que contem a classe do NaïveBayes
/------------------------------------------------------------------------*/
#ifndef CLASSE_H
#define CLASSE_H
#include <vector>
#include <fstream>

class MatrixVector
{
    	public:
    	/*! Atributo utilizado para guardar valores em vetores. */
		std::vector<double> vetor;
};

//-------------------------------------------------------------------------
class Matrix2d
{
    	public:
    	/*! Atributo utilizado para guardar valores em vetores. */
		std::vector<double> camada;

	Matrix2d(): camada(100,0) {} //inicializa o vetor com 100 posições e valor 0
};

//-------------------------------------------------------------------------
class Matrix3d: public MatrixVector
{
    	public:
    	/*! Atributo utilizado para guardar valores em matriz. */
		MatrixVector camada[100];
};

//-------------------------------------------------------------------------
class RNA: private Matrix3d, private Matrix2d
{
	public:

    	/*!
   	 * Construtor.
   	 */
    	RNA(){std::cout<<"Criando rede Neural"<<std::endl;   	srand (time(NULL));};
    	/*! Destrutor */
    	//~NaiveBayes();

        /*!Método que realiza o NaïveBayes.
	* @param entrada - Entrada para os valores desejado.
	* @param filename - Nome do arquivo que deseja carregar.
	* @param numCol - Número de colunas do arquivo.
        * @return Não retorna valores.*/
	void Rna( std::string fileName, int numCol);

	void runLearnedFile(std::string fileName, std::string fileLearned);

	void runTraining(std::string fileName, std::string fileLearned);

	void runNewTraining(std::string fileName, std::string fileLearned, double Won, int numNeuroniosN, double alphaN, double nN, int numCamadaEscondidaN, unsigned int numEntradasN, unsigned int iteracoes, unsigned int numSaidasN);

	private:

        /*!Método para abrir o arquivo e carregar para a matriz.
	* @param matrix[] - Matriz para carregar os valores da tabela.
	* @param filename - Nome do arquivo que deseja carregar.
	* @param numCol - Número de colunas do arquivo.
        * @return Não retorna valores. */
	unsigned int openFiletoGetQvalueVector(MatrixVector matrix[], unsigned int numCol, std::string fileName);

	double NeuronioArtificial(std::vector<double> X, std::vector<double> Wi, double Wo, double &net);

	double fnet(double net);

	double dfnet(double net);

	void RnaTraining(std::vector<double> Xinput, std::vector<double> target, Matrix3d W[], MatrixVector DWm[], MatrixVector Ws[], Matrix3d dWtemp[], MatrixVector dWStemp[], double Wo, int numNeuronios, int numNeuroFinal, std::vector<double> &Saida, double &erro);

	void RnaLearned(std::vector<double> Xinput, Matrix3d W[], MatrixVector Ws[], double Wo, int numNeuronios, int numNeuroFinal, std::vector<double> &Saida);

	void saveValue( Matrix3d W[], MatrixVector Ws[], double Wo, int numNeuronios, int numNeuroFinal, double alphaF, double nF, int numCamadaEscondidaF, unsigned int numEntradasF, unsigned int iteracoes, std::string fileName);

	void openValueW( Matrix3d W[], MatrixVector Ws[], double &Wo, int &numNeuronios, int &numNeuroFinal, double &alphaF, double &nF, int &numCamadaEscondidaF, unsigned int &numEntradasF, unsigned int &iteracoes, std::string fileName);

	void openParametros( double &Wo, int &numNeuronios, int &numNeuroFinal, double &alphaF, double &nF, int &numCamadaEscondidaF, unsigned int &numEntradasF, unsigned int &iteracoes, unsigned int &numSaidas, std::string fileName);

	double alpha;
	double n;
	int numCamadaEscondida;
	unsigned int numEntradas;
};
 
/// \example classe.cpp
 
#endif
