/*-------------------------------------------------------------------------
***************************************************************************
* @file classe.cpp
* @author Isaac Jesus da Silva - Centro Universitário da FEI
* @version V0.0.1
* @created 04/11/2014
* @Modified 06/11/2014
* @e-mail isaac25silva@yahoo.com.br
***************************************************************************
Arquivo fonte que contem a implementação da Rede Neural Artificial
/------------------------------------------------------------------------*/

#include <iostream>
#include "classe.h"
#include <cmath>
#include <iterator>
#include <fstream>
#include <vector>
#include <time.h>       /* time */
#include <stdlib.h>     /* srand, rand */

using namespace std;

void RNA::Rna( std::string fileName, int numCol)
{

	n = 0.30;
	alpha = 0.2;
	numEntradas = 256;
	unsigned int numSaidas = 10;
	numCamadaEscondida=3;
	double erro, erroTotal=0;
	double erroDesejado=0.0000001;
	double Wo=0; //bias

    	unsigned int contador=0;
	int numNeuronios=1000;
	int numNeuroFinal=numSaidas;
	
	MatrixVector ts[10]; //Saída desejada do treinamento

	MatrixVector matrix[numCol]; //Cria objetos do tipo matrix para inserir a tabela

	openFiletoGetQvalueVector(matrix, numCol, fileName); //le do arquivo a tabela

	//Carrega os parametros do arquivo
	//openParametros( Wo, numNeuronios, numNeuroFinal,alpha,n, numCamadaEscondida, numEntradas, contador, numSaidas,"Learned.dat");

	MatrixVector Xinput[matrix[0].vetor.size()]; // Entradas do neurônio
	std::vector<double> XinputValor(matrix[0].vetor.size()); //Guarda os valores da classe em relação aos dados
	//std::vector<double> ts(numNeuroFinal); //Guarda os valores da classe em relação aos dados
	std::vector<double> Saida(numNeuroFinal); //
	Matrix3d W[numNeuronios]; //W das camadas escondidas
	MatrixVector Ws[numNeuroFinal]; //W da  camada de Saída
	Matrix3d dWtemp[numNeuronios]; //copia dos W das camadas escondidas
	MatrixVector dWStemp[numNeuroFinal]; //W da  camada de Saída

	MatrixVector DWm[numNeuronios]; //delta W da camada intermediaria e ultima camada

	//--------Carrega as variaveis com valores zeros--------
	for(int y=0; y<numNeuronios;y++)
	{
	    for(int x=0; x<numNeuronios;x++)
	    {
		DWm[y].vetor.push_back(0);
		for(int z=0; z<numCamadaEscondida; z++)
		    dWtemp[y].camada[z].vetor.push_back(0);
	    }
	    for(int x=0; x<numNeuroFinal;x++)
	    {
		dWStemp[x].vetor.push_back(0);
	    }
	}
	//-------------------------------------------------------
	

    for(int y=0; y<10; y++)
    {
	for(int x=0; x<10; x++)
	{
	    if(x==y)
		ts[y].vetor.push_back(1);
	    else
		ts[y].vetor.push_back(0);
	   cout<<ts[y].vetor[x]<<endl;
	}
	cout<<endl;
    }


//	//imprime toda a matriz de valores
//	for(int y=0; y<matrix[0].vetor.size(); y++)
//	    for(int x=0; x<numCol; x++)
//		cout<<matrix[x].vetor[y]<<" ";
//	cout<<endl;

	// Transpoe os dados de entrada-----------------
	for(int y=0; y<matrix[0].vetor.size(); y++)
	    for(int x=0; x<numCol; x++)
	    {
		if(x==0)
		    XinputValor[y] = matrix[x].vetor[y]; //Valor referente a saida desejada  -> tm
		else
		    Xinput[y].vetor.push_back(matrix[x].vetor[y]);
	    }

//	//imprime toda a matriz de valores
//	for(int y=0; y<matrix[0].vetor.size(); y++)
//	    for(int x=0; x<numCol; x++)
//		cout<<Xinput[y].vetor[x]<<" ";
//	cout<<endl;
//		cout<<"tamanho = "<<Xinput[0].vetor.size()<<endl;

//	for(int y=0; y<matrix[0].vetor.size(); y++)
//		cout<<ts[y]<<" ";


	    //Valores randomicos de Win-------------------------------------
	    for(int z=0; z<numCamadaEscondida; z++)
		for(int y=0; y<numNeuronios; y++)
	    	    for(int x=0; x<numNeuronios; x++)// Varre todo o vetor Xinput de entrada
			W[y].camada[z].vetor.push_back( -1+(rand()/(double)RAND_MAX)*2); //Gera pesos randomicos para W
	    //-------------------------------------------------------------


	    //Valores randomicos de Ws-------------------------------------
		for(int y=0; y<numNeuroFinal; y++)
	    	    for(int x=0; x<numNeuronios; x++) // Varre todo o vetor Sj de entrada
			Ws[y].vetor.push_back( -1+(rand()/(double)RAND_MAX)*2); //Gera pesos randomicos para Ws
	    //-------------------------------------------------------------

    int valor;
    do
    {
    	for(unsigned int amostra=0; amostra<matrix[0].vetor.size(); amostra++)
   	{
	//cout<<XinputValor[amostra]<<endl;

	//while(1){}
	
	//cout<<"Valor = "<<XinputValor[amostra]<<endl;
	    valor = XinputValor[amostra];
//	    cout<<"Valor = "<<valor<<endl;
//	    return;

	    RnaTraining(Xinput[amostra].vetor , ts[valor].vetor, W, Ws, DWm, dWtemp, dWStemp, Wo, numNeuronios, numNeuroFinal, Saida, erro);
	//cout<<endl;
	erroTotal+=abs(erro);
	    if(amostra%200 == 0)
	    {
            	cout<<"Valor = "<<valor<<endl;
	    	for(int x=0; x<numNeuroFinal; x++)
	   	    cout<<"S["<<x<<"] = "<<Saida[x]<<endl;
		cout<<"Erro Total = "<<erroTotal<<" Amostra = "<<amostra<<" Contador = "<<contador<<endl<<endl;
		erroTotal=0;
	    }

    	}
	contador++;
	saveValue( W, Ws, Wo, numNeuronios, numNeuroFinal, alpha, n, numCamadaEscondida, numEntradas, contador, "Learned.dat");

    //}while(abs(erroTotal)>erroDesejado);
    }while(1);


	//openValueW( W, Ws, Wo, numNeuronios, numNeuroFinal, alpha, n, numCamadaEscondida, numEntradas, contador, "Learned.dat");

////----codigo dos 4 padroes de numeros-------------------------------------
//	//cout<<"Valor = "<<XinputValor[amostra]<<endl;
//    int x=0;
//    do
//    {
//	x++;
//	erroTotal=0;
//	for(int amostra=0; amostra<4; amostra++)
//	{
//	    RnaTraining(Xinput[amostra].vetor , ts[amostra].vetor, W, Ws, DWm, dWtemp, dWStemp, Wo, numNeuronios, numNeuroFinal, Saida, erro);
//	    erroTotal+=abs(erro);
//	}
//	if(x%1000==0)
//	{
//	    erroTotal=0;
//	    for(int amostra=0; amostra<4; amostra++)
//	    {
//	    	RnaLearned(Xinput[amostra].vetor, W, Ws, Wo, numNeuronios, numNeuroFinal, Saida);
//	    	for(int x=0; x<numSaidas; x++)
//	            cout<<amostra<<" Saida["<<x<<"] = "<<Saida[x]<<endl;
//		erroTotal+=abs(erro);
//	    }
//	    	cout<<"Erro Total = "<<erroTotal<<endl<<endl;
//	}

//    }while(abs(erroTotal)>erroDesejado);

//	for(int amostra=0; amostra<4; amostra++)
//	{
//	    RnaLearned(Xinput[amostra].vetor, W, Ws, Wo, numNeuronios, numNeuroFinal, Saida);
//	    for(int x=0; x<numSaidas; x++)
//	        cout<<"Saida["<<x<<"] = "<<Saida[x]<<endl;
//	    cout<<endl;
//	}
////------------------------------------------------------------------------------------------

}

//===============================================================================================
//-------------Neurônio Perceptron -----------------------------------------------
double RNA::NeuronioArtificial(	std::vector<double> Xi, std::vector<double> Wi, double Wo, double &net)
{

	double Sj=0;
	net=0;
	//realiza o somatorio -> net = \sum_{i=0}^N (W_i*X_i)   onde: N = numero de entradas
	for(int i=0; i<numEntradas; i++) // Varre todo o vetor de entrada
	{
		net += Wi[i]*Xi[i];
		//cout<<"net = "<<net<<endl;
	}
	net = net + Wo;

	Sj = fnet(net);

	return Sj;

}
//===============================================================================================
//------------ Função sigmoid ou tanh ----------- -----------------------------------------------
//===============================================================================================
//As formulas estão na página 168 e 169 no capitulo 4 da 2° edição no livro do Haykin
double RNA::fnet(double net)
{
	//return 1/(1 + exp(-net)); //   f(net) = 1/(1 + exp(-a*net))  formula do livro Haykin
	return 1*tanh(0.1*net); // a*tanh(b*net) formula do livro Haykin
}
//===============================================================================================
//------------ Derivada da função sigmoid ou tanh -----------------------------------------------
//===============================================================================================
double RNA::dfnet(double net)
{
	//return fnet(net)*(1-fnet(net)); // f'(net) = a*f(net)*(1-f(net)) formula do livro Haykin
	return (0.1/1)*(1 - fnet(net))*(1 + fnet(net)); // f'(net) = (b/a)*(a-f(net))*(a+f(net)) formula do livro Haykin
}
//===============================================================================================
//-------------função sigmoid-------------------- -----------------------------------------------
void RNA::RnaTraining(std::vector<double> Xinput, std::vector<double> target, Matrix3d W[], MatrixVector Ws[], MatrixVector DWm[], Matrix3d dWtemp[], MatrixVector dWStemp[], double Wo, int numNeuronios, int numNeuroFinal, std::vector<double> &Saida, double &erro)
{

	erro=0;
	Matrix2d Sj[numNeuronios]; //vetor da primeira camada escondida
	MatrixVector Sjt[numCamadaEscondida]; //vetor da primeira camada escondida
	Matrix2d netj[numNeuronios]; //net da primeira camada escondida

	std::vector<double> Ss(numNeuroFinal); //vetor da ultima camada
	std::vector<double> netS(numNeuroFinal); //net da ultima camada

	std::vector<double> ein(numNeuronios); //erro da primeira camada-
	MatrixVector DWin[numNeuronios]; //delta W da primeira camada

	MatrixVector ec[numCamadaEscondida]; //erro da ultima camada


	//Inicio do treinamento--------------------------------------------------------------------------

	//----Realiza o Feed-Foward nas camadas --------------------------------------------------------------

	//Primeira camadas escondida da rede neural ---------------------------------------------
	for(int x=0; x<numNeuronios; x++) //Alimenta todos os neurônios da primeira camada escondida
	{
		Sj[x].camada[0] = NeuronioArtificial( Xinput, W[x].camada[0].vetor, Wo, netj[x].camada[0]);
		//cout<<"Sj["<<x<<"] = "<<Sj[x]<<endl;
		Sjt[0].vetor.push_back(Sj[x].camada[0]); //trasnpõe o Sj
	}
	//--------------------------------------------------------------------------------------

	if(numCamadaEscondida>1)
	{
	   for(int y=1; y<numCamadaEscondida; y++)
	   {
	   //Camadas escondidas da rede neural-----------------------------------------------------
	   	for(int x=0; x<numNeuronios; x++) //Alimenta todos os neurônios da primeira camada escondida
	   	{
			Sj[x].camada[y] = NeuronioArtificial( Sjt[y-1].vetor, W[x].camada[y].vetor, Wo, netj[x].camada[y]);
			//cout<<"Sj["<<x<<"] = "<<Sj[x]<<endl;
			Sjt[y].vetor.push_back(Sj[x].camada[y]); //trasnpõe o Sj
	   	}
	   }
	}
	//--------------------------------------------------------------------------------------

	//Ultima camada da rede neural ---------------------------------------------------------
	for(int x=0; x<numNeuroFinal; x++) //Alimenta todos os neurônios da primeira camada escondida
	{
	    Ss[x] = NeuronioArtificial( Sjt[numCamadaEscondida-1].vetor, Ws[x].vetor, Wo, netS[x]); //Entra com Sj qué é a saída da ultima camada
		//cout<<"Ss["<<x<<"] = "<<Ss[x]<<endl;
	}
	Saida = Ss;
	//--------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------


	//----Realiza o Feed-Backward nas camadas --------------------------------------------------------

	// Feed-Backward camada de saida -----------------------------------------------------
	for(int y=0; y<numNeuroFinal; y++)
	{
	    //------- em = (tm - sm)*f'(net) ------------
		ec[0].vetor.push_back((target[y] - Ss[y])*dfnet(netS[y]));
	    //-------------------------------------------

	    for(int x=0; x<numNeuronios;x++)
	    {
	    //------- DWm = n*em*Sk + alpha*Dwm(t-1)---------------
		DWm[y].vetor[x] = (n*ec[0].vetor[y]*Sjt[numCamadaEscondida-1].vetor[x] + alpha*dWStemp[y].vetor[x]);
	    //-------------------------------------------

	    //------- Ws = Ws + DWs -------------------------------
		Ws[y].vetor[x] +=DWm[y].vetor[x];
	    //-----------------------------------------------------
	    }
	    //copia do W e Ws para usar no calculo de momento -----------------
	    dWStemp[y].vetor = DWm[y].vetor;

	    erro += ec[0].vetor[y];
	}
	//-----------------------------------------------------------------


	// Feed-Backward das camadas intermediárias ---------------------------------------
     if(numCamadaEscondida>1)
     {
	int contE = 1;
	for(int x=0; x<numNeuronios ;x++)
	{
	     //------- ein = (\sum{em*Wm})*f'(net) -----------------
	    double somaEW = 0;
	    for(int y=0; y<numNeuroFinal; y++)
		somaEW += ec[0].vetor[y]*Ws[y].vetor[x];
	    ec[1].vetor.push_back(somaEW*dfnet(netj[x].camada[numCamadaEscondida-1]));
	}
	for(int x=0; x<numNeuronios ;x++)
	{
	    //------- DWin = n*ein*Xin  + alpha*Dwtemp(t-1)------------------------------
	    for(int z=0; z<numNeuronios;z++)
	    {
	    	    DWm[x].vetor[z] = (n*ec[1].vetor[x]*Sjt[numCamadaEscondida-2].vetor[z] + alpha*dWtemp[x].camada[numCamadaEscondida-1].vetor[z]);
	    }
	    //---------------------------------------------------------------------------

	    //------- W = W + DW ------------------------------
	    for(int z=0; z<numNeuronios;z++)
		W[x].camada[numCamadaEscondida-1].vetor[z] += DWm[x].vetor[z];
	    //-------------------------------------------------------

	    //copia do W para usar no calculo de momento -----------------
	    dWtemp[x].camada[numCamadaEscondida-1].vetor = DWm[x].vetor;
	    //cout<<DWin[x].vetor[0]<< endl;
	}

	if(numCamadaEscondida>2)
	{
	  for(int c=(numCamadaEscondida-2); c>0; c--)
	  {
	    for(int x=0; x<numNeuronios ;x++)
	    {
	     	//------- ein = (\sum{em*Wm})*f'(net) -----------------
	    	double somaEW = 0;
	    	for(int y=0; y<numNeuronios; y++)
		    somaEW += ec[contE].vetor[y]*W[y].camada[c].vetor[x];
	    	ec[contE+1].vetor.push_back(somaEW*dfnet(netj[x].camada[c]));
	    }
	    for(int x=0; x<numNeuronios ;x++)
	    {
	   	 //------- DWin = n*ein*Xin  + alpha*Dwtemp(t-1)------------------------------
	    	for(int z=0; z<numNeuronios;z++)
	    	{
	    	    DWm[x].vetor[z] = (n*ec[contE+1].vetor[x]*Sjt[c-1].vetor[z] + alpha*dWtemp[x].camada[c].vetor[z]);
	    	}
	    	//---------------------------------------------------------------------------

	    	//------- W = W + DW ------------------------------
	    	for(int z=0; z<numNeuronios;z++)
		    W[x].camada[c].vetor[z] += DWm[x].vetor[z];
	    	//-------------------------------------------------------

	    	//copia do W para usar no calculo de momento -----------------
	    	dWtemp[x].camada[c].vetor = DWm[x].vetor;
	    	//cout<<DWin[x].vetor[0]<< endl;
	    }
	    contE++;
	  }
	  contE--;
	}
	

	//-----------------------------------------------------------------

	// Feed-Backward camada entrada------------------------------------------------
	for(int x=0; x<numNeuronios ;x++)
	{
	     //------- ein = (\sum{em*Dwm})*f'(net) -----------------
	    double somaEW = 0;
	    for(int y=0; y<numNeuronios; y++)
		somaEW = somaEW + ec[contE].vetor[y]*W[y].camada[1].vetor[x];
	    ein[x] = somaEW*dfnet(netj[x].camada[0]);

	    //------- DWin = n*ein*Xin  + alpha*Dwtemp(t-1)------------------------------
	    for(int z=0; z<Xinput.size();z++)
	    {
		if(dWtemp[x].vetor.size()==0) // se o vetor DWtemp ainda não foi copiado
	    	    DWin[x].vetor.push_back(n*ein[x]*Xinput[z]);
		else
	    	    DWin[x].vetor.push_back(n*ein[x]*Xinput[z] + alpha*dWtemp[x].camada[0].vetor[z]);
	    //}
	    //---------------------------------------------------------------------------

	    //------- Win = Win + DWin ------------------------------
	    //for(int z=0; z<Xinput.size();z++)
		W[x].camada[0].vetor[z] = W[x].camada[0].vetor[z] + DWin[x].vetor[z];
	    //-------------------------------------------------------
	    }
	    //copia do W para usar no calculo de momento -----------------
	    dWtemp[x].camada[0].vetor = DWin[x].vetor;
	    //cout<<DWin[x].vetor[0]<< endl;
	}	
	//-----------------------------------------------------------------


     }
     else
     {
	// Feed-Backward camada entrada------------------------------------------------
	for(int x=0; x<numNeuronios ;x++)
	{
	     //------- ein = (\sum{em*Dwm})*f'(net) -----------------
	    double somaEW = 0;
	    for(int y=0; y<numNeuroFinal; y++)
		somaEW = somaEW + ec[0].vetor[y]*Ws[y].vetor[x];
	    ein[x] = somaEW*dfnet(netj[x].camada[0]);


	    for(int z=0; z<Xinput.size();z++)
	    {
	    //------- DWin = n*ein*Xin  + alpha*Dwtemp(t-1)------------------------------
	    	DWin[x].vetor.push_back(n*ein[x]*Xinput[z] + alpha*dWtemp[x].camada[0].vetor[z]);
	    //---------------------------------------------------------------------------

	    //------- Win = Win + DWin ------------------------------
		W[x].camada[0].vetor[z] = W[x].camada[0].vetor[z] + DWin[x].vetor[z];
	    //-------------------------------------------------------
	    }
	    //copia do W para usar no calculo de momento -----------------
	    dWtemp[x].camada[0].vetor = DWin[x].vetor;
	    //cout<<DWin[x].vetor[0]<< endl;
	}	
	//-----------------------------------------------------------------
     }

}
//===============================================================================================
//----------------------------------------------- -----------------------------------------------
void RNA::RnaLearned(std::vector<double> Xinput, Matrix3d W[], MatrixVector Ws[], double Wo, int numNeuronios, int numNeuroFinal, std::vector<double> &Saida)
{

	Matrix2d Sj[numNeuronios]; //vetor da primeira camada escondida
	MatrixVector Sjt[numCamadaEscondida]; //vetor da primeira camada escondida
	Matrix2d netj[numNeuronios]; //net da primeira camada escondida

	std::vector<double> Ss(numNeuroFinal); //vetor da ultima camada
	std::vector<double> netS(numNeuroFinal); //net da ultima camada

	std::vector<double> ein(numNeuronios); //erro da primeira camada-
	MatrixVector DWin[numNeuronios]; //delta W da primeira camada

	MatrixVector ec[numCamadaEscondida]; //erro da ultima camada


	//Inicio do treinamento--------------------------------------------------------------------------

	//----Realiza o Feed-Foward nas camadas --------------------------------------------------------------

	//Primeira camadas escondida da rede neural ---------------------------------------------
	for(int x=0; x<numNeuronios; x++) //Alimenta todos os neurônios da primeira camada escondida
	{
		Sj[x].camada[0] = NeuronioArtificial( Xinput, W[x].camada[0].vetor, Wo, netj[x].camada[0]);
		//cout<<"Sj["<<x<<"] = "<<Sj[x]<<endl;
		Sjt[0].vetor.push_back(Sj[x].camada[0]); //trasnpõe o Sj
	}
	//--------------------------------------------------------------------------------------

	if(numCamadaEscondida>1)
	{
	   for(int y=1; y<numCamadaEscondida; y++)
	   {
	   //Camadas escondidas da rede neural-----------------------------------------------------
	   	for(int x=0; x<numNeuronios; x++) //Alimenta todos os neurônios da primeira camada escondida
	   	{
			Sj[x].camada[y] = NeuronioArtificial( Sjt[y-1].vetor, W[x].camada[y].vetor, Wo, netj[x].camada[y]);
			//cout<<"Sj["<<x<<"] = "<<Sj[x]<<endl;
			Sjt[y].vetor.push_back(Sj[x].camada[y]); //trasnpõe o Sj
	   	}
	   }
	}
	//--------------------------------------------------------------------------------------

	//Ultima camada da rede neural ---------------------------------------------------------
	for(int x=0; x<numNeuroFinal; x++) //Alimenta todos os neurônios da primeira camada escondida
	{
	    Ss[x] = NeuronioArtificial( Sjt[numCamadaEscondida-1].vetor, Ws[x].vetor, Wo, netS[x]); //Entra com Sj qué é a saída da ultima camada
		//cout<<"Ss["<<x<<"] = "<<Ss[x]<<endl;
	}
	Saida = Ss;
	//--------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------


}
//===============================================================================================
//---------------- Executa um treinamento novo --------------------------------------------------
//===============================================================================================
//----------------------------------------------- -----------------------------------------------
void RNA::runNewTraining(std::string fileName, std::string fileLearned,double Won, int numNeuroniosN, double alphaN, double nN, int numCamadaEscondidaN, unsigned int numEntradasN, unsigned int iteracoes, unsigned int numSaidasN)
{
	cout<<"Inicializando o treinamento ..."<<endl;

	//carrega as variáveis----------------
	n = nN;
	alpha = alphaN;
	numEntradas = numEntradasN;
	unsigned int numSaidas = numSaidasN;
	double erro, erroTotal=0;
	double erroDesejado=0.0000001;
	double Wo=Won; //bias
	numCamadaEscondida=numCamadaEscondidaN;
    	unsigned int contador=0;
	int numNeuronios = numNeuroniosN;
	int numNeuroFinal=numSaidas;
	
	MatrixVector ts[numSaidas]; //Saída desejada do treinamento

	MatrixVector matrix[numEntradas+1]; //Cria objetos do tipo matrix para inserir a tabela vinda do arquivo

	cout<<"Carregando os dados de Treinamento do Arquivo ..."<<endl;
	openFiletoGetQvalueVector(matrix, numEntradas+1, fileName); //le do arquivo a tabela

	MatrixVector Xinput[matrix[0].vetor.size()]; // Entradas do neurônio
	std::vector<double> XinputValor(matrix[0].vetor.size()); //Guarda os valores da classe em relação aos dados
	//std::vector<double> ts(numNeuroFinal); //Guarda os valores da classe em relação aos dados
	std::vector<double> Saida(numNeuroFinal); //
	Matrix3d W[numNeuronios]; //W das camadas escondidas
	MatrixVector Ws[numNeuroFinal]; //W da  camada de Saída
	Matrix3d dWtemp[numNeuronios]; //copia dos W das camadas escondidas
	MatrixVector dWStemp[numNeuroFinal]; //W da  camada de Saída

	MatrixVector DWm[numNeuronios]; //delta W da camada intermediaria e ultima camada

	//--------Carrega as variaveis com valores zeros--------
	for(int y=0; y<numNeuronios;y++)
	{
	    for(int x=0; x<numNeuronios;x++)
	    {
		DWm[y].vetor.push_back(0);
		for(int z=0; z<numCamadaEscondida; z++)
		    dWtemp[y].camada[z].vetor.push_back(0);
	    }
	    for(int x=0; x<numNeuroFinal;x++)
	    {
		dWStemp[x].vetor.push_back(0);
	    }
	}
	//-------------------------------------------------------
	

    for(int y=0; y<10; y++)
    {
	for(int x=0; x<10; x++)
	{
	    if(x==y)
		ts[y].vetor.push_back(1);
	    else
		ts[y].vetor.push_back(-1);
	   cout<<ts[y].vetor[x]<<endl;
	}
	cout<<endl;
    }

	// Transpoe os dados de entrada-----------------
	for(int y=0; y<matrix[0].vetor.size(); y++)
	    for(int x=0; x<(numEntradas+1); x++)
	    {
		if(x==0)
		    XinputValor[y] = matrix[x].vetor[y]; //Valor referente a saida desejada  -> tm
		else
		    Xinput[y].vetor.push_back(matrix[x].vetor[y]);
	    }
	//--------------------------------------------------



	    //Valores randomicos de Win-------------------------------------
	    for(int z=0; z<numCamadaEscondida; z++)
		for(int y=0; y<numNeuronios; y++)
	    	    for(int x=0; x<numNeuronios; x++)// Varre todo o vetor Xinput de entrada
			W[y].camada[z].vetor.push_back( -1+(rand()/(double)RAND_MAX)*2); //Gera pesos randomicos para W
	    //-------------------------------------------------------------


	    //Valores randomicos de Ws-------------------------------------
		for(int y=0; y<numNeuroFinal; y++)
	    	    for(int x=0; x<numNeuronios; x++) // Varre todo o vetor Sj de entrada
			Ws[y].vetor.push_back( -1+(rand()/(double)RAND_MAX)*2); //Gera pesos randomicos para Ws
	    //-------------------------------------------------------------


	cout<<"Iniciando o treinamento da RNA"<<endl;

    int valor;
    do
    {
    	for(unsigned int amostra=0; amostra<matrix[0].vetor.size(); amostra++)
   	{
	//cout<<XinputValor[amostra]<<endl;

	//while(1){}
	
	//cout<<"Valor = "<<XinputValor[amostra]<<endl;
	    valor = XinputValor[amostra];
//	    cout<<"Valor = "<<valor<<endl;
//	    return;

	    RnaTraining(Xinput[amostra].vetor , ts[valor].vetor, W, Ws, DWm, dWtemp, dWStemp, Wo, numNeuronios, numNeuroFinal, Saida, erro);
	//cout<<endl;
	erroTotal+=abs(erro);
	    if(amostra%200 == 0)
	    {
            	cout<<"Valor = "<<valor<<endl;
	    	for(int x=0; x<numNeuroFinal; x++)
		{
		    //imprime os valores de treinamento e verifica quantos acertos
		    if(Saida[x] > 0.5)
		    {
	           	 cout<<"\e[1;94mSaida["<<x<<"] = "<<Saida[x]<<"\e[0m"<<endl; //imprime colorido
		    }
		    else
		    {
		    	if( Saida[x]> 0.1 && Saida[x] <= 0.5)
	           	    cout<<"\e[1;91mSaida["<<x<<"] = "<<Saida[x]<<"\e[0m"<<endl; //imprime colorido
		    	else
			    cout<<"Saida["<<x<<"] = "<<Saida[x]<<endl;
		    }
		}
		cout<<"Erro Total = "<<erroTotal<<" Amostra = "<<amostra<<" Contador = "<<contador<<endl<<endl;
		erroTotal=0;
		saveValue( W, Ws, Wo, numNeuronios, numNeuroFinal, alpha, n, numCamadaEscondida, numEntradas, contador, fileLearned);
	    }

    	}
	contador++;

    //}while(abs(erroTotal)>erroDesejado);
    }while(contador<iteracoes);

	cout<<"Treinamento Finalizado"<<endl;
}
//===============================================================================================
//---------------- Executa o treinamento a partir de valores W de um treinamento anterior -------
//===============================================================================================
//----------------------------------------------- -----------------------------------------------
void RNA::runTraining(std::string fileName, std::string fileLearned)
{
	cout<<"Inicializando o treinamento ..."<<endl;

	unsigned int numSaidas = 10;
	double erro, erroTotal=0;
	double erroDesejado=0.0000001;
	double Wo=0; //bias

    	unsigned int contador=0;
	int numNeuronios;
	int numNeuroFinal=numSaidas;

	//Carrega os parametros do arquivo
	openParametros( Wo, numNeuronios, numNeuroFinal,alpha,n, numCamadaEscondida, numEntradas, contador, numSaidas, fileLearned);

	MatrixVector ts[numSaidas]; //Saída desejada do treinamento

	MatrixVector matrix[numEntradas+1]; //Cria objetos do tipo matrix para inserir a tabela

	cout<<"Carregando os dados de Treinamento do Arquivo ..."<<endl;
	openFiletoGetQvalueVector(matrix, numEntradas+1, fileName); //le do arquivo a tabela

	MatrixVector Xinput[matrix[0].vetor.size()]; // Entradas do neurônio
	std::vector<double> XinputValor(matrix[0].vetor.size()); //Guarda os valores da classe em relação aos dados
	//std::vector<double> ts(numNeuroFinal); //Guarda os valores da classe em relação aos dados
	std::vector<double> Saida(numNeuroFinal); //
	Matrix3d W[numNeuronios]; //W das camadas escondidas
	MatrixVector Ws[numNeuroFinal]; //W da  camada de Saída
	Matrix3d dWtemp[numNeuronios]; //copia dos W das camadas escondidas
	MatrixVector dWStemp[numNeuroFinal]; //W da  camada de Saída

	MatrixVector DWm[numNeuronios]; //delta W da camada intermediaria e ultima camada

	//--------Carrega as variaveis com valores zeros--------
	for(int y=0; y<numNeuronios;y++)
	{
	    for(int x=0; x<numNeuronios;x++)
	    {
		DWm[y].vetor.push_back(0);
		for(int z=0; z<numCamadaEscondida; z++)
		    dWtemp[y].camada[z].vetor.push_back(0);
	    }
	    for(int x=0; x<numNeuroFinal;x++)
	    {
		dWStemp[x].vetor.push_back(0);
	    }
	}
	//-------------------------------------------------------
	

    for(int y=0; y<10; y++)
    {
	for(int x=0; x<10; x++)
	{
	    if(x==y)
		ts[y].vetor.push_back(1);
	    else
		ts[y].vetor.push_back(-1);
	   cout<<ts[y].vetor[x]<<endl;
	}
	cout<<endl;
    }

	// Transpoe os dados de entrada-----------------
	for(int y=0; y<matrix[0].vetor.size(); y++)
	    for(int x=0; x<257; x++)
	    {
		if(x==0)
		    XinputValor[y] = matrix[x].vetor[y]; //Valor referente a saida desejada  -> tm
		else
		    Xinput[y].vetor.push_back(matrix[x].vetor[y]);
	    }
	//--------------------------------------------------


	    //Valores de Win-------------------------------------
	    for(int z=0; z<numCamadaEscondida; z++)
		for(int y=0; y<numNeuronios; y++)
	    	    for(int x=0; x<numNeuronios; x++)// Varre todo o vetor Xinput de entrada
			W[y].camada[z].vetor.push_back(0); //Gera os W
	    //-------------------------------------------------------------

	    //Valores de Ws-------------------------------------
		for(int y=0; y<numNeuroFinal; y++)
	    	    for(int x=0; x<numNeuronios; x++) // Varre todo o vetor Sj de entrada
			Ws[y].vetor.push_back(0); //Gera os Ws
	    //-------------------------------------------------------------

	cout<<"Carregando do Arquivo os paramentros W aprendido ..."<<endl;
	openValueW( W, Ws, Wo, numNeuronios, numNeuroFinal, alpha, n, numCamadaEscondida, numEntradas, contador, fileLearned);

	cout<<"Iniciando o treinamento da RNA"<<endl;


    int valor;
    do
    {
    	for(unsigned int amostra=0; amostra<matrix[0].vetor.size(); amostra++)
   	{
	//cout<<XinputValor[amostra]<<endl;

	//while(1){}
	
	//cout<<"Valor = "<<XinputValor[amostra]<<endl;
	    valor = XinputValor[amostra];
//	    cout<<"Valor = "<<valor<<endl;
//	    return;

	    RnaTraining(Xinput[amostra].vetor , ts[valor].vetor, W, Ws, DWm, dWtemp, dWStemp, Wo, numNeuronios, numNeuroFinal, Saida, erro);
	//cout<<endl;
	erroTotal+=abs(erro);
	    if(amostra%200 == 0)
	    {
            	cout<<"Valor = "<<valor<<endl;
	    	for(int x=0; x<numNeuroFinal; x++)
		{
		    //imprime os valores de treinamento e verifica quantos acertos
		    if(Saida[x] > 0.5)
		    {
	           	 cout<<"\e[1;94mSaida["<<x<<"] = "<<Saida[x]<<"\e[0m"<<endl; //imprime colorido
		    }
		    else
		    {
		    	if( Saida[x]> 0.1 && Saida[x] <= 0.5)
	           	    cout<<"\e[1;91mSaida["<<x<<"] = "<<Saida[x]<<"\e[0m"<<endl; //imprime colorido
		    	else
			    cout<<"Saida["<<x<<"] = "<<Saida[x]<<endl;
		    }
		}
		cout<<"Erro Total = "<<erroTotal<<" Amostra = "<<amostra<<" Iterações = "<<contador<<endl<<endl;
		erroTotal=0;
		saveValue( W, Ws, Wo, numNeuronios, numNeuroFinal, alpha, n, numCamadaEscondida, numEntradas, contador, fileLearned);
	    }

    	}
	contador++;
    //}while(abs(erroTotal)>erroDesejado);
    }while(1);

	cout<<"Treinamento Finalizado"<<endl;
}
//==============================================================================================
//==============================================================================================
//--------------------Executa dados na RNA com a rede já treinada--------------------------------
//===============================================================================================
//----------------------------------------------- -----------------------------------------------
void RNA::runLearnedFile(std::string fileName, std::string fileLearned)
{
	cout<<"Inicializando o teste ..."<<endl;

	unsigned int numSaidas = 10;
	double erro, erroTotal=0;
	double erroDesejado=0.0000001;
	double Wo=0; //bias

    	unsigned int contador=0;
	int numNeuronios;
	int numNeuroFinal=numSaidas;

	unsigned int acertos=0, errosV=0;
	
	//MatrixVector ts[10]; //Saída desejada do treinamento

	//Carrega os parametros do arquivo
	openParametros( Wo, numNeuronios, numNeuroFinal,alpha,n, numCamadaEscondida, numEntradas, contador, numSaidas,fileLearned);

	MatrixVector matrix[numEntradas+1]; //Cria objetos do tipo matrix para inserir a tabela

	cout<<"Carregando do Arquivo os dados de Entrada ..."<<endl;
	openFiletoGetQvalueVector(matrix, numEntradas+1, fileName); //le do arquivo a tabela

	MatrixVector Xinput[matrix[0].vetor.size()]; // Entradas do neurônio
	std::vector<double> XinputValor(matrix[0].vetor.size()); //Guarda os valores da classe em relação aos dados
	std::vector<double> Saida(numNeuroFinal); //
	Matrix3d W[numNeuronios]; //W das camadas escondidas
	MatrixVector Ws[numNeuroFinal]; //W da  camada de Saída

	// Transpoe os dados de entrada-----------------
	for(int y=0; y<matrix[0].vetor.size(); y++)
	    for(int x=0; x<257; x++)
	    {
		if(x==0)
		    XinputValor[y] = matrix[x].vetor[y]; //Valor referente a saida desejada  -> tm
		else
		    Xinput[y].vetor.push_back(matrix[x].vetor[y]);
	    }
	//--------------------------------------------------



	    //Valores de Win-------------------------------------
	    for(int z=0; z<numCamadaEscondida; z++)
		for(int y=0; y<numNeuronios; y++)
	    	    for(int x=0; x<numNeuronios; x++)// Varre todo o vetor Xinput de entrada
			W[y].camada[z].vetor.push_back(0); //Gera os W
	    //-------------------------------------------------------------

	    //Valores de Ws-------------------------------------
		for(int y=0; y<numNeuroFinal; y++)
	    	    for(int x=0; x<numNeuronios; x++) // Varre todo o vetor Sj de entrada
			Ws[y].vetor.push_back(0); //Gera os Ws
	    //-------------------------------------------------------------

	cout<<"Carregando do Arquivo os paramentros W aprendido ..."<<endl;
	openValueW( W, Ws, Wo, numNeuronios, numNeuroFinal, alpha, n, numCamadaEscondida, numEntradas, contador, fileLearned);

	cout<<"Iniciando passagem de dados pela RNA"<<endl;
	for(int amostra=0; amostra<matrix[0].vetor.size(); amostra++)
	{
	    RnaLearned(Xinput[amostra].vetor, W, Ws, Wo, numNeuronios, numNeuroFinal, Saida);
	    //cout<<"Valor desejado = "<<XinputValor[amostra]<<" | Amostra num = "<<amostra<<endl;
	    for(int x=0; x<numSaidas; x++)
	    {
		//imprime os valores de treinamento e verifica quantos acertos
		if(Saida[x] > 0.5)
		{
	            //cout<<"\e[1;94mSaida["<<x<<"] = "<<Saida[x]<<"\e[0m"<<endl; //imprime colorido
		    if(x==XinputValor[amostra])
			acertos++;
		    else
			errosV++;
		}
		else
		{
//		    if( Saida[x]> 0.1 && Saida[x] <= 0.5)
//	           	cout<<"\e[1;91mSaida["<<x<<"] = "<<Saida[x]<<"\e[0m"<<endl; //imprime colorido

//		    else
//			cout<<"Saida["<<x<<"] = "<<Saida[x]<<endl;
		}
	    }
	    //cout<<endl;
	    //getchar();
	}
	cout<<"Teste finalizado"<<endl;
	cout<<"Houve "<<acertos<<" acertos em "<<matrix[0].vetor.size()<<" amostras"<<endl;
	cout<<"Houve "<<errosV<<" conflitos em "<<matrix[0].vetor.size()<<" amostras"<<endl;
	cout<<"Rede com "<<numNeuronios<<" por camada escondida com "<<numCamadaEscondida<<" camadas"<<endl;
	cout<<"Taxa de aprendizado de "<<n<<" e com taxa de momento de "<<alpha<<endl;
	cout<<"Foi executado "<<contador<<" treinamentos"<<endl;

}
//==============================================================================================
//--------------------Salva os valores do treinamento no arquivo--------------------------------
void RNA::saveValue( Matrix3d W[], MatrixVector Ws[], double Wo, int numNeuronios, int numNeuroFinal, double alphaF, double nF, int numCamadaEscondidaF, unsigned int numEntradasF, unsigned int iteracoes, std::string fileName)
{
    std::string separator = " "; // Use blank as default separator between single features
    std::fstream File;

    File.open( fileName, std::ios::out);
    if (File.good() && File.is_open())
    {
//        for (int i = 0; i < 161; ++i)
//        {
//            for (int j = 0; j < 2; ++j)
//                    File << qvalues[i][j] << separator;
//        }
	File << iteracoes << separator;
	File << alphaF << separator;
	File << nF << separator;
	File << numCamadaEscondidaF << separator;
	File << numEntradasF << separator;
	File << numNeuroFinal << separator;
	File << numNeuronios << separator;
	File << Wo << separator;
        File << std::endl;

	    for(int z=0; z<numCamadaEscondidaF; z++)
		for(int y=0; y<numNeuronios; y++)
	    	    for(int x=0; x<numNeuronios; x++)// Varre todo o vetor Xinput de entrada
			File << W[y].camada[z].vetor[x]<< separator;

        File << std::endl;

	    //Valores de Ws-------------------------------------
		for(int y=0; y<numNeuroFinal; y++)
	    	    for(int x=0; x<numNeuronios; x++) // Varre todo o vetor Sj de entrada
			File << Ws[y].vetor[x]<< separator;
	    //-------------------------------------------------------------
        File.flush();
        File.close();
    }
    else
	printf("Erro ao Salvar o arquivo\n");
}
//===============================================================================================
//-------------Abre o arquivo que contem a tabela -----------------------------------------------
void RNA::openValueW( Matrix3d W[], MatrixVector Ws[], double &Wo, int &numNeuronios, int &numNeuroFinal, double &alphaF, double &nF, int &numCamadaEscondidaF, unsigned int &numEntradasF, unsigned int &iteracoes, std::string fileName)
{
    const char *fileName1;
    fileName1 = fileName.c_str();

	std::ifstream File(fileName1);

	if (File.fail())
	{
		cout << "file opening failed" << endl;
		system("PAUSE");
		exit (1);
	}

	std::istream_iterator<double> start(File), end; //inicio e fim do arquivo
	std::vector<double> vectorTemp(start, end); //Vetor temporário para carregar todo o arquivo
	//std::vector<double> titulo(numCol); //Vetor que carrega o cabeçalho

	unsigned int inc=0;

	iteracoes = vectorTemp[0];
	alphaF = vectorTemp[1];
	nF = vectorTemp[2];
	numCamadaEscondidaF  = vectorTemp[3];
	numEntradasF  = vectorTemp[4];
	numNeuroFinal  = vectorTemp[5];
	numNeuronios  = vectorTemp[6];
	Wo  = vectorTemp[7];

	inc=8;
	    for(int z=0; z<numCamadaEscondidaF; z++)
		for(int y=0; y<numNeuronios; y++)
	    	    for(int x=0; x<numNeuronios; x++)// Varre todo o vetor Xinput de entrada
		    {
			W[y].camada[z].vetor[x] = vectorTemp[inc];
			inc++;
		    }

	    //Valores de Ws-------------------------------------
	    for(int y=0; y<numNeuroFinal; y++)
	    	    for(int x=0; x<numNeuronios; x++) // Varre todo o vetor Sj de entrada
		    {
			Ws[y].vetor[x] = vectorTemp[inc];
			inc++;
		    }
	    //-------------------------------------------------------------

}
//===============================================================================================
//-------------Abre o arquivo que contem a tabela -----------------------------------------------
void RNA::openParametros( double &Wo, int &numNeuronios, int &numNeuroFinal, double &alphaF, double &nF, int &numCamadaEscondidaF, unsigned int &numEntradasF, unsigned int &iteracoes, unsigned int &numSaidas, std::string fileName)
{
    const char *fileName1;
    fileName1 = fileName.c_str();

	std::ifstream File(fileName1);

	if (File.fail())
	{
		cout << "file opening failed" << endl;
		system("PAUSE");
		exit (1);
	}

	std::istream_iterator<double> start(File), end; //inicio e fim do arquivo
	std::vector<double> vectorTemp(start, end); //Vetor temporário para carregar todo o arquivo
	//std::vector<double> titulo(numCol); //Vetor que carrega o cabeçalho

	unsigned int inc=0;

	iteracoes = vectorTemp[0];
	alphaF = vectorTemp[1];
	nF = vectorTemp[2];
	numCamadaEscondidaF  = vectorTemp[3];
	numEntradasF  = vectorTemp[4];
	numNeuroFinal  = vectorTemp[5];
	numNeuronios  = vectorTemp[6];
	Wo  = vectorTemp[7];

	numSaidas = numNeuroFinal;

}
//==============================================================================================
//===============================================================================================
//-------------Abre o arquivo que contem a tabela -----------------------------------------------
unsigned int RNA::openFiletoGetQvalueVector(MatrixVector matrix[], unsigned int numCol, std::string fileName)
{
    const char *fileName1;
    //std::string fileName = "ex1_1.txt";
    //std::string fileName = "ex2_1.txt";
    fileName1 = fileName.c_str();

	std::ifstream File(fileName1);
	std::istream_iterator<double> start(File), end; //inicio e fim do arquivo
	std::vector<double> vectorTemp(start, end); //Vetor temporário para carregar todo o arquivo
	//std::vector<double> titulo(numCol); //Vetor que carrega o cabeçalho

	int y=0;
        for (int i = 0; i < vectorTemp.size(); i=i+numCol)
        {
		for(int x=0; x<numCol; x++) //Carrega o vetor
		{
			matrix[x].vetor.push_back(vectorTemp[i+x]);
			//cout<<matrix[x].vetor[y]<<"\t";
		}
                //std::cout<<std::endl;
		y++;
        }

	return 0;
}
//==============================================================================================


