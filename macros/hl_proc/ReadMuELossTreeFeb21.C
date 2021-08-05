#define ReadMuELossTreeFeb21_cxx
#include "ReadMuELossTreeFeb21.h"
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TF1.h>
#include <TProfile.h>
#include <TStyle.h>
#include <TCanvas.h>
#include "TROOT.h"
#include "TMath.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"
#include <math.h>
#include "TRandom.h"
#include "TRandom3.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>

//VARIABILI STATICHE
static const int nmaxcells = 1000; // max size of clusters
//static const int nmaxcells = 200; 
static double largenumber = 10000000;
static const int ND=26;//27
static double e3d[32][32][50]; //energy matrix
static double emin = 0.1;
static int ncell=0;
static int vx[nmaxcells];
static int vy[nmaxcells];
static int vz[nmaxcells];
static int vr[nmaxcells];
static double ve[nmaxcells];
static double eclu=0;

// LIKELIHOOD FIT VARIABLES
static double xxx[5];
static double zzz[5];
static double eee[5];
static double variance[5];
static double xmax[50];
static double ymax[50];
static double zmax[50];
static double emax_[50];
static double hiterr = 10.;
static double sign;

//CLUSTERING RICORSIVO
void cluster3d (int ncellmin) {
  if (ncell>nmaxcells-1) return;
  if (ncellmin>nmaxcells-1) return;
  //ncellmin: numero di celle già presenti nel cluster
  bool exit = false;
  
  for (int i=ncellmin-1; i<nmaxcells && !exit; i++) {
    if (ncell>nmaxcells-1) return;
    //cerco l'indice z massimo e minimo del cluster per poi calcolarne l'ampiezza in z
    double maxz = -1;
    double minz = 100000;
    for (int j=0; j<i; j++) {    
      if(vz[j] < minz) minz = vz[j];
      if(vz[j] > maxz) maxz = vz[j];
    }
    
    //se il cluster ha un ampiezza inferiore a 5 in z
    //if(maxz - minz < 5){
    //nel caso l'ampiezza sia di 3 celle o meno
    //if (maxz - minz < 4){
    if (vz[i]>0) {
      //se trova un cella con energia superiore a quella minima
      if (e3d[vx[i]][vy[i]][vz[i]-1]>emin) {
	//inserisce le coordinata e l'energia della cella nel cluster
	vz[ncell]=vz[i]-1;
	vx[ncell]=vx[i];
	vy[ncell]=vy[i];
	double ethis = e3d[vx[ncell]][vy[ncell]][vz[ncell]];
	ve[ncell]=ethis;
	// aggiungo all'energia totale del cluster l'energia della cella
	eclu+=ethis;
	//resetto la cella così da non aggiungerla in un altro cluster
	e3d[vx[ncell]][vy[ncell]][vz[ncell]]=0;
	//passa alla prossima cella
	ncell++;
      }
    }
    if (ncell>=nmaxcells) return;
    // look in tower ahead in z
    if (vz[i]<49) {
      if (e3d[vx[i]][vy[i]][vz[i]+1]>emin) {
	vz[ncell]=vz[i]+1;
	vx[ncell]=vx[i];
	vy[ncell]=vy[i];
	double ethis = e3d[vx[ncell]][vy[ncell]][vz[ncell]];
	ve[ncell]=ethis;
	eclu+=ethis;
	e3d[vx[ncell]][vy[ncell]][vz[ncell]]=0;
	ncell++;
      }
    }
    if (ncell>=nmaxcells) return;
    //}
    //caso "limite" di profondità pari a 4
    /*
      else if(maxz-minz == 4){
      if(vz[i]>0 && vz[i]<49){
      //salvo le energie delle celle anteriore e posteriore in z
      double e1 = e3d[vx[i]][vy[i]][vz[i]-1];
      double e2 = e3d[vx[i]][vy[i]][vz[i]+1];
      //se almeno una delle due è più alta della soglia
      if(e2 > emin || e1 > emin){
      //aggiungo al cluster solo uella più alta
      if(e2 > e1) {
      vz[ncell]=vz[i]+1;
      vx[ncell]=vx[i];
      vy[ncell]=vy[i];
      double ethis = e3d[vx[ncell]][vy[ncell]][vz[ncell]];
      ve[ncell]=ethis;
      eclu+=ethis;
      e3d[vx[ncell]][vy[ncell]][vz[ncell]]=0;
      ncell++;
      }
      else {
      vz[ncell]=vz[i]-1;
      vx[ncell]=vx[i];
      vy[ncell]=vy[i];
      double ethis = e3d[vx[ncell]][vy[ncell]][vz[ncell]];
      ve[ncell]=ethis;
      eclu+=ethis;
      e3d[vx[ncell]][vy[ncell]][vz[ncell]]=0;
      ncell++;
      }
      }
      }
      }*/		
    //}
    // look left
    if (vx[i]>0) {
      if (e3d[vx[i]-1][vy[i]][vz[i]]>emin) {
	vz[ncell]=vz[i];
	vx[ncell]=vx[i]-1;
	vy[ncell]=vy[i];
	double ethis = e3d[vx[ncell]][vy[ncell]][vz[ncell]];
	ve[ncell]=ethis;
	eclu+=ethis;
	e3d[vx[ncell]][vy[ncell]][vz[ncell]]=0;
	ncell++;
      }      
    }
    if (ncell>=nmaxcells) return;
    // look right
    if (vx[i]<31) {
      if (e3d[vx[i]+1][vy[i]][vz[i]]>emin) {
	vz[ncell]=vz[i];
	vx[ncell]=vx[i]+1;
	vy[ncell]=vy[i];
	double ethis = e3d[vx[ncell]][vy[ncell]][vz[ncell]];
	ve[ncell]=ethis;
	eclu+=ethis;
	e3d[vx[ncell]][vy[ncell]][vz[ncell]]=0;
	ncell++;
      }      
    }
    if (ncell>=nmaxcells) return;
    // look up
    if (vy[i]<31) {
      if (e3d[vx[i]][vy[i]+1][vz[i]]>emin) {
	vz[ncell]=vz[i];
	vx[ncell]=vx[i];
	vy[ncell]=vy[i]+1;
	double ethis = e3d[vx[ncell]][vy[ncell]][vz[ncell]];
	ve[ncell]=ethis;
	eclu+=ethis;
	e3d[vx[ncell]][vy[ncell]][vz[ncell]]=0;
	ncell++;
      }      
    }
    if (ncell>=nmaxcells) return;
    // look down
    if (vy[i]>0) {
      if (e3d[vx[i]][vy[i]-1][vz[i]]>emin) {
	vz[ncell]=vz[i];
	vx[ncell]=vx[i];
	vy[ncell]=vy[i]-1;
	double ethis = e3d[vx[ncell]][vy[ncell]][vz[ncell]];
	ve[ncell]=ethis;
	eclu+=ethis;
	e3d[vx[ncell]][vy[ncell]][vz[ncell]]=0;
	ncell++;
      }      
    }
    if (ncell>=nmaxcells) return;
    if (ncell==ncellmin) exit=true; // could not find neighbors, exiting
    if (ncell>=nmaxcells-7) exit=true; // do not overflow
  }
  if (ncell>nmaxcells-2) return;
  if (ncell>ncellmin) cluster3d(ncellmin+1);
  return;
}


//Verosimiglianza per la singola osservazione (con 2 punti)
double logprobx (double xm, double zm, double em, double x0, double r) {
	double xpred = x0+sign*(r-sqrt(r*r-zm*zm));
	double logp = -0.5*pow((xpred-xm)/(hiterr/em),2);
	if (logp!=logp) return 0.;
	return logp;
}

/*
double logprobx2 (double xm, double zm, double em, double var,double x0, double r) {
	double xpred = x0+sign*sqrt(r*r-zm*zm); //
	double logp = -0.5*pow((xm - xpred)/(hiterr/em/var),2);
	if (logp!=logp) return 0.;
	return logp;
}*/


//Verosimiglianza (prodotto delle singole osservazioni)
extern "C" void Likelihood (int& npar, double* grad, double& fval, double* xval, int flag) {
	double x0 = xval[0];
	double r  = xval[1];
	double logL  = 0.;
	for (int i=1; i<50; i++ ) {
		if (emax_[i]>0) {
			double logp = logprobx(xmax[i],zmax[i],emax_[i],x0,r);
			logL = logL - logp;
		}
	}
	fval = logL;
}


/*
//Verosimiglianza (prodotto di 3+ osservazioni)
extern "C" void Likelihood2 (int& npar, double* grad, double& fval, double* xval, int flag) {
	double x0 = xval[0];
	double r  = xval[1];
	double logL  = 0.;
	for (int i=1; i<3; i++) {
		if (eee[i]>0) {
			double logp = logprobx2(xxx[i],zzz[i],eee[i],variance[i],x0,r);
			logL = logL + logp;
		}
	}
	fval = logL;
}
*/
//MAIN LOOP
void ReadMuELossTreeFeb21::Loop(int sample) {
	
  delete gRandom;
  TRandom3 * myRNG = new TRandom3();
  
  TChain * chain = new TChain("B4","");
  
  std::stringstream sstr;
  sstr << sample;
  string asciipath = "/lustre/cmswork/dorigo/MuELoss/FixedE/ascii/";
  string rootpath  = "/lustre/cmswork/dorigo/MuELoss/FixedE/root/";	
  string asciifile = asciipath  + sstr.str() + ".asc";
  TString rootfile  = rootpath   + sstr.str() + ".root";
  
  ofstream results;
  results.open(asciifile);
  
  //if (sample==0) chain->Add("/lustre/cmswork/dorigo/MuELoss/Feb21/root/0.root");
  
  chain->Add(rootfile);
  TTree * tree=chain;
  Init(tree);
  
  // Variables
  // results <<  "eabove " << 	//0
  
  /*"rmom " <<  	//2
    "rmom1 " << 	//3
    "r2m " << 		//4
    "r2m1 "<< 		//5
    "r2m2 "<< 		//6
    "r2m3 "<< 		//7
    "r2m4 "<< 		//8
    "r2m5 "<< 		//9
    "nclu "<< 		//10
    "ncellmax "<< 	//11
    "eclumax "<< 	//12
    "maxdepthx "<<  //13
    "maxdepthy "<<  //14
    "maxdepthz  "<<  //15
    "epercell " <<  //16
    "avgcells "<< 	//17
    "nclu1 " << 	//18
    "ncellmax1 "<<  //19
    "eclumax1 "<< 	//20
    "epercell1 "<<  //21
    "avgcells1 "<<  //22
    "xmom "<< 		//23
    "ymom "<< 		//24
    "emeas "<< 	    //25
    "maxsum " << 	//26
    "eb_1 "<< 		//27
    "ea_1 "<< 		//28
    //"eb_2 "<< 		//29
    //"ea_2 "<< 		//30
    //"eb_3 "<< 		//31
    //"ea_3 "<< 		//32
    //"eb_4 "<< 		//33
    //"ea_4 "<< 		//34
    "true_energy"  	//35
  */

  //initialize plots
  TH1D * Diff = new TH1D ("Diff","",100, -1000,1000);
  TH1D * Diff_fit = new TH1D ("Diff_fit","",100, -1000,1000);
  TH2D * DiffE= new TH2D ("DiffE","",50, -1000., 1000., 50, -1000.,1000.);
  TH1D * F_R = new TH1D ( "F_R", "", 100, 0., 100.);
  TH1D * DE[10];
  char name[30];
  for (int ie=0; ie<10; ie++) {
    sprintf (name,"DE%d",ie);
    DE[ie] = new TH1D (name,name,20,-2.,2.);
  }
  
  //load number of entries
  if (fChain == 0) return;
  Long64_t nentries = fChain->GetEntriesFast();
  int nprocess = nentries;
  //cout << "Entries of chain = " << nentries << endl;
  //nprocess = 50;
  
  //print progress
  Long64_t nbytes = 0, nb = 0;
  int block = nprocess/50;
  char progress[53] = "[--10%--20%--30%--40%--50%--60%--70%--80%--90%-100%]";
  cout<< "  Processing data:  " << progress[0];
  int currchar = 1;
  
  
	
  
  
  double emeas;
  //empty containers + initialization
  double data[ND]; //features
  double alpha[4];
  double nsplit = 4;
  double navg=0; //sample size
  double avgdata[ND]; //average
  double avgdata2[ND]; //average squared
  double mina[ND]; //minimum
  double maxa[ND]; //maximum
  double ebelow_[8];
  //initialize
  for (int i=0; i<ND; i++) { 
    avgdata[i]=0; 
    avgdata2[i]=0;
    mina[i]=largenumber;
    maxa[i]=-largenumber;
  }
  for(int a = 0; a < nsplit; a++){
    if(a == 0) alpha[0] = .01;
    else alpha[a] = alpha[a-1] + .01;
  }
  // START OF LOOP
  for (Long64_t jentry=0; jentry<nprocess; jentry++) {
    // print progress and load file
    if (jentry%block==0) {
      cout << progress[currchar];
      currchar++;
    }
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   
    nbytes += nb;
    
    // initialize features vector
    for (int i=0; i<ND; i++) { data[i]=0; };
    for (int i=0; i<(nsplit*2); i++) { ebelow_[i]=0;}
    // center xy of cell hit by muon
    double xmu, ymu;
    int ix = (int)(true_x/3.75);
    if (true_x>0) xmu = (double)ix*3.75+1.875;
    else xmu = (double)ix*3.75-1.875;
    int iy = (int)(true_y/3.75);
    if (true_y>0) ymu = (double)iy*3.75+1.875;
    else ymu = (double)iy*3.75-1.875;
    // energia totale sopra e sotto la soglia emin
    double ebelow = 0.; //data[0]
    double eabove = 0.;	//data[1]
    //data[2] momento trasverso mancante
    double sumex=0;
    double sumey=0;
    double sume=0;
    //data[3] momento trasverso mancante per e > emin
    double sumex1=0;
    double sumey1=0;
    double sume1=0;
    //data[4]-data[9]
    double sume2r=0;
    double sumez1=0;
    double sume2rz1=0;
    double sumez2=0;
    double sume2rz2=0;
    double sumez3=0;
    double sume2rz3=0;
    double sumez4=0;
    double sume2rz4=0;
    double sumez5=0;
    double sume2rz5=0;

    int charge = -1;
    if (isAntiMuon==1) charge=1;

    //initialize
    for (int i=0; i<50; i++) {
      emax_[i]=0;
      xmax[i]=0;
      zmax[i]=0;
    }
    for (int i=0; i<32; i++) {
      for (int j=0; j<32; j++) {
	for (int k=0; k<50; k++) {
	  e3d[i][j][k]=0.;
	}
      }
    }
    // ciclo sulla matrice3d
    for (UInt_t j=0; j<rechit_energy->size(); ++j) {
      //energy from tree
      double e = rechit_energy->at(j);
      //coordinates from tree
      double x = rechit_x->at(j);
      double y = rechit_y->at(j);
      double z = rechit_z->at(j);
      //integer coordinates
      int ix = (int)((x+60.)/3.75);
      int iy = (int)((y+60.)/3.75);
      int iz = (int)(z/40.);
      //fill energy3d matrix
      e3d[ix][iy][iz]=e;
      double xrel = x-true_x; //distance from center 
      double yrel = y-true_y;
      double w = exp(-fabs(y-ymu)/50.); // posso modificare questo
      emax_[iz]+=e*w; 
      xmax[iz]+=x*e*w; //baricentro energetico per ogni z
      zmax[iz]+=z*e*w; 
      sumex += xrel*e; //E_x per data[2] (vedi articolo)
      sumey += yrel*e; //E_y per data[2] (vedi articolo)
      sume += e;
      if (e>emin) {
	sumex1 += xrel*e; //E_x per data[3] (vedi articolo)
	sumey1 += yrel*e; //E_y per data[3] (vedi articolo)
	sume1 += e;
      }
      //euclidean distance
      double dist =sqrt(xrel*xrel+yrel*yrel); 
      //necessario per data[4] (vedi articolo)
      sume2r += dist*dist*e; //numeratore di data[4]
      // divido il detector lungo z [0-400][400-800][800-1200][1200-1600][1600+]
      // necessarii per data[5]-...-data[9]
      if (z<400) { //somma lungo z < 400
	sume2rz1+=dist*dist*e; //numeratore
	sumez1+=e; //denominatore
      } else if (z<800) {
	sume2rz2+=dist*dist*e;
	sumez2+=e;
      } else if (z<1200) {
	sume2rz3+=dist*dist*e;
	sumez3+=e;
      } else if (z<1600) {
	sume2rz4+=dist*dist*e;
	sumez4+=e;
      } else {
	sume2rz5+=dist*dist*e;
	sumez5+=e;
      }
      // ebelow e eabove + binned eabove
      if (e<=emin) {
	ebelow+=e;
	for(int i = 0; i < nsplit; i++){
	  if(e <= alpha[i]) ebelow_[i*2] +=e;
	  else ebelow_[i*2+1]+= e;
	}
      } else {
	eabove+=e;
      }
    }
    
    for (int i=0; i<50; i++) {
      if (emax_[i]>0) {
	xmax[i]=xmax[i]/emax_[i];
	zmax[i]=zmax[i]/emax_[i];
      }
    }
    for (int i=0; i<5; i++) {
      xxx[i]=0;
      zzz[i]=0;
      eee[i]=0;
    }
    
    double Bfield = 2.;
    double xmed1=0;
    double zmed1=0;
    double xmed2=0;
    double zmed2=0;
    double xmed3=0;
    double zmed3=0;
    double xmed4=0;
    double zmed4=0;
    double xmed5=0;
    double zmed5=0;
    double sum_e=0;
    double sumn=0;
    
    //primo punto
    for (int i=0; i<25; i++) { 
      xmed1 = xmed1+xmax[i]*emax_[i];
      zmed1 = zmed1+zmax[i]*emax_[i];
      sum_e += emax_[i];
      sumn++;
    }
    if (sum_e>0) xmed1 = xmed1/sum_e;
    if (sum_e>0) zmed1 = zmed1/sum_e;
    eee[0] = sum_e;
    
    //secondo punto
    sum_e=0; 
    sumn=0; 
    for (int i=25; i<50; i++) { 
      xmed2 = xmed2+xmax[i]*emax_[i];
      zmed2 = zmed2+zmax[i]*emax_[i];
      sum_e += emax_[i];
      sumn++;
    }
    if (sum_e>0) xmed2 = xmed2/sum_e;
    if (sum_e>0) zmed2 = zmed2/sum_e;
    eee[1] =sum_e;
    
    //terzo punto
    /*
      sum_e=0; 
      sumn=0; 
      for (int i=40; i<50; i++) { 
      xmed3 = xmed3+xmax[i]*emax_[i];
      zmed3 = zmed3+zmax[i]*emax_[i];
      sum_e += emax_[i];
      sumn++;
      }
      if (sum_e>0) xmed3 = xmed3/sum_e;
      if (sum_e>0) zmed3 = zmed3/sum_e;
      eee[2] =sum_e;
    */
    /*
      sum_e=0; 
      sumn=0; 
      for (int i=45; i<50; i++) { 
      xmed4 = xmed4+xmax[i]*emax_[i];
      zmed4 = zmed4+zmax[i]*emax_[i];
      sum_e += emax_[i];
      sumn++;
      }
      
      if (sum_e>0) xmed4 = xmed4/sum_e;
      if (sum_e>0) zmed4 = zmed4/sum_e;
      eee[3] =sum_e;
      sum_e=0; 
      sumn=0; 
      for (int i=45; i<50; i++) { 
      xmed5 = xmed5+xmax[i]*emax_[i];
      zmed5 = zmed5+zmax[i]*emax_[i];
      sum_e += emax_[i];
      sumn++;
      }
      if (sum_e>0) xmed5 = xmed5/sum_e;
      if (sum_e>0) zmed5 = zmed5/sum_e;
      eee[3] =sum_e;
    */
    
    xxx[0] = xmed1;
    zzz[0] = zmed1;
    xxx[1] = xmed2;
    zzz[1] = zmed2;
    //xxx[2] = xmed3;
    //zzz[2] = zmed3;
    //xxx[3] = xmed4;
    //zzz[3] = zmed4;
    //xxx[4] = xmed5;
    //zzz[4] = zmed5;
    
    emeas=0;
    if (xmed2!=xmed1) {//raggio della circonferenza
      //raggio della circonferenza con 2 punti solo
      double dz = zmed2-zmed1;
      if (dz==0) dz = 1000.; // half calo depth
      double dx = xmed2-xmed1;
      double x0 = (zmed2*zmed2-zmed1*zmed1+xmed2*xmed2-xmed1*xmed1)/(2*(xmed2-xmed1));
      double r = sqrt(xmed1*xmed1-2*xmed1*x0+x0*x0+zmed1*zmed1); //stimo raggio
      if (xmed1>xmed2) r = - r;
      //stima energia misurata
      emeas = 0.3*Bfield*r/1000.;
      //plot
      if (true_energy<1000) {
	Diff->Fill(sign*true_energy-emeas);
      }
      for (int ie=0; ie<10; ie++) {
	double ethisbin=ie*100.+50.;
	if (fabs(ethisbin-true_energy)<50.) {
	  DE[ie]->Fill((sign*true_energy-emeas)/true_energy);
	}
      }
      DiffE->Fill(sign*true_energy,emeas);
    }
    
    //curvature fit
    sign = 1.;
    if (isMuon==1) sign = -1.;
    bool dofit=false;
    double regre=0.;
    if (dofit) {
      TMinuit rmin (1);
      int iflag = 0; // You can use this for selection
      double arglis[4];
      double Start[2]; 
      double Step[2]    = {1., 1000.};
      double Min[2]     = {-200., -100000000.};
      double Max[2]     = { 200.,  100000000.};
      TString parnam[2] = { "x0",  "r" };
      double a[2], err[2], pmin, pmax;
      int ivar;
      rmin.SetFCN (Likelihood);
      // Main initialization member function for MINUIT
      rmin.mninit (5,6,7);
      // Parameters needed to be unambiguous with MINOS
      // arglis[0]=2;
      arglis[0] = -1; // HACK -1; // quiet mode
      rmin.mnexcm ("SET PRI", arglis, 1, iflag);
      rmin.mnexcm ("SET NOW", arglis, 1, iflag);
      arglis[0] = 2; // 2
      // Sets the strategy to be used in calculating first and second derivatives 
      // and in certain minimization methods. 1 is default
      rmin.mnexcm ("SET STR", arglis, 1, iflag);
      // Set fit parameters
      // ------------------
      Start[0] = true_x; // myRNG->Uniform(-60.,60.); // true_x;
      Start[1] = myRNG->Uniform(0.5*1000.*true_energy/(0.3*Bfield),2*1000.*true_energy/(0.3*Bfield)); // 1000000; curvature expected given energy, in 2 tesla field, in mm
      rmin.mnparm (0, parnam[0],  Start[0], Step[0], Min[0], Max[0], iflag);
      rmin.mnparm (1, parnam[1],  Start[1], Step[1], Min[1], Max[1], iflag);
      // Instructs Minuit to call subroutine FCN with the value of IFLAG
      arglis[0] = 1;
      rmin.mnexcm ("call fcn", arglis, 1, iflag); // command executed normally	  
      arglis[0] = 0;
      rmin.mnexcm ("mini", arglis, 1, iflag);
      
      // Read results
      // ------------
      rmin.mnpout (0, parnam[0], a[0], err[0], pmin, pmax, ivar);
      rmin.mnpout (1, parnam[1], a[1], err[1], pmin, pmax, ivar);
      double result1[2];
      double result1_err[2];
      for (int res=0; res<2; res++) {
	result1[res] = a[res];
	result1_err[res] = err[res];
      }
      
      // Fit 2 - Set fit parameters
      // --------------------------
      rmin.SetFCN (Likelihood);
      // Main initialization member function for MINUIT
      rmin.mninit (5,6,7);
      // Parameters needed to be unambiguous with MINOS
      // arglis[0]=2;
      arglis[0] = -1; // HACK -1; // quiet mode
      rmin.mnexcm ("SET PRI", arglis, 1, iflag);
      rmin.mnexcm ("SET NOW", arglis, 1, iflag);
      arglis[0] = 2; // 2
      // Sets the strategy to be used in calculating first and second derivatives 
      // and in certain minimization methods. 1 is default
      rmin.mnexcm ("SET STR", arglis, 1, iflag);
      Start[0] = true_x; // myRNG->Uniform(-60.,60.); // true_x;
      Start[1] = -myRNG->Uniform(0.5*1000.*true_energy/(0.3*Bfield),2*1000.*true_energy/(0.3*Bfield)); // 1000000; curvature expected given energy, in 2 tesla field, in mm
      rmin.mnparm (0, parnam[0],  Start[0], Step[0], Min[0], Max[0], iflag);
      rmin.mnparm (1, parnam[1],  Start[1], Step[1], Min[1], Max[1], iflag);
      // Instructs Minuit to call subroutine FCN with the value of IFLAG
      arglis[0] = 1;
      rmin.mnexcm ("call fcn", arglis, 1, iflag); // command executed normally	  
      arglis[0] = 0;
      rmin.mnexcm ("mini", arglis, 1, iflag);
      
      // Read results
      // ------------
      rmin.mnpout (0, parnam[0], a[0], err[0], pmin, pmax, ivar);
      rmin.mnpout (1, parnam[1], a[1], err[1], pmin, pmax, ivar);
      double result2[2];
      double result2_err[2];
      for (int res=0; res<2; res++) {
	result2[res] = a[res];
	result2_err[res] = err[res];
      }
      
      // cout results
      regre = result1[1]/1000.*0.3*Bfield;
      if (fabs(result2[1])<fabs(result1[1])) {
	regre = result2[1]/1000.*0.3*Bfield;
      }
      if (true_energy<500) {
	cout << "True energy = " << sign*true_energy << ", regressed = " << regre << endl;
      }
      if (true_energy<1000) Diff_fit->Fill(true_energy-regre);
      
    } // if dofit
    
    data[0] = eabove; //data[0]
    //data[1] = ebelow; //data[1]
    
    //data[2]-data[3]
    double rmom = sqrt(sumex*sumex+sumey*sumey);
    if (sume>0) rmom=rmom/sume;
    data[1] = rmom;
    double rmom1 = sqrt(sumex1*sumex1+sumey1*sumey1);
    if (sume1>0) rmom1=rmom1/sume1;
    data[2] = rmom1;
    
    //data[4]-...-data[9]
    double r2m=sume2r;
    if (sume>0) r2m=r2m/sume;
    data[3] = r2m;
    double r2m1 = sume2rz1;
    if (sumez1>0) r2m1=r2m1/sumez1;
    data[4] = r2m1;
    double r2m2 = sume2rz2;
    if (sumez2>0) r2m2=r2m2/sumez2;
    data[5] = r2m2;
    double r2m3 = sume2rz3;
    if (sumez3>0) r2m3=r2m3/sumez3;
    data[6] = r2m3;
    double r2m4 = sume2rz4;
    if (sumez4>0) r2m4=r2m4/sumez4;
    data[7] = r2m4;
    double r2m5 = sume2rz5;
    if (sumez5>0) r2m5=r2m5/sumez5;
    data[8] = r2m5;
    
    //data[16]-data[17]
    double xmom = sumex+sumex1; //E_x + E_x_1
    double ymom = sumey+sumey1; //E_y + E_y_1
    
    
    //3d clustering lungo la traiettoria
    double nclu=0; //numero di cluster
    double eclumax=0; //energia massima totale tra i cluster
    double ncellmax=0; //numero massimo di celle tra i cluster
    
    int imax=-1;
    // Now loop on highest-E tower along muon direction (ixm, iym) and search for clusters,
    // every time removing towers from array
    int ixm = (int)((true_x+60.)/3.75);
    int iym = (int)((true_y+60.)/3.75);
    
    double avge =0;
    double avgcells=0;
    
    double maxdepthx = 0;
    double maxdepthy = 0;
    double maxdepthz = 0;
    
    //cout << "entry:"<< jentry <<endl;
    do {
      
      //initialize
      for (int index=0; index<nmaxcells; index++) { 
	//nmaxcells = max number of cell per cluster
	vz[index]=0;
	vx[index]=0;
	vy[index]=0;
	ve[index]=0.;
      }
      
      ncell=0;
      eclu=0;
      double emax = emin;
      imax = -1;
      
      for (int iz=0; iz<50; iz++) { //scorre lungo il piano trasverso
	//trova l'energia massima e la coordinata corrispondente in z
	if (e3d[ixm][iym][iz] > emax) { 
	  imax = iz;
	  emax = e3d[ixm][iym][iz];
	}
      }
      
      //cout << imax << endl;
      
      if (imax>-1) { //se esiste un'energia sopra la soglia
	//cout << nclu;
	nclu++; //aggiunge un cluster
	//cout << " " << nclu << endl;
	// salva le coordinate e l'energia della priam cella
	// alla prima iterazione ncell=0 contiene la prima cella del cluster
	vz[ncell]=imax;
	vx[ncell]=ixm;
	vy[ncell]=iym;
	ve[ncell]=e3d[ixm][iym][imax];
	eclu+=ve[ncell]; //aggiunge l'energia all'energia totale del cluster
	e3d[ixm][iym][imax]=0.; //resetta la cella così non entra nel cluster successivo
	ncell++; //cerca la seconda cella
	
	cluster3d(ncell); //cerca il cluster intorno alla cella
	
      }
      //energia massima totale e numero massimo di celle
      avge += eclu;
      avgcells += ncell;
      if (ncell>nmaxcells) ncell= nmaxcells; // protect from overflows

      if (eclu>eclumax) eclumax=eclu;
      //if (eclu < eclumin) eclumin = eclu;
      if (ncell > ncellmax) ncellmax=ncell;
      
      double max = 0;
      double min = 1000000;
      for (int j=0; j<ncell; j++) {    
	if(vx[j] < min) min = vx[j];
	if(vx[j] > max) max = vx[j];
      }
      double depthx = max - min;
      
      max = 0;
      min = 1000000;
      for (int j=0; j<ncell; j++) {    
	if(vy[j] < min) min = vy[j];
	if(vy[j] > max) max = vy[j];
      }
      double depthy = max - min;
      
      max = 0;
      min = 1000000;
      for (int j=0; j<ncell && j<nmaxcells ; j++) {    
	if(vz[j] < min) min = vz[j];
	if(vz[j] > max) max = vz[j];
      }
      double depthz = max - min;
      
      if(depthx > maxdepthx) maxdepthx = depthx;
      if(depthy > maxdepthy) maxdepthy = depthy;
      if(depthz > maxdepthz) maxdepthz = depthz;
      
    } while (imax>-1); 
    //cout << endl;
    
    data[9] = nclu;
    data[10] = ncellmax;
    data[11] = eclumax;
    
    data[12] = maxdepthx;
    data[13] = maxdepthy;
    data[14] = maxdepthz;
    
    if (nclu == 0){
      data[15] = 0;
      data[16] = 0;
    } else {
      if (ncellmax != 0) data[15] = eclumax/ncellmax;
      else data[15] =0;
      data[16] = avgcells/nclu;
    }
    
    // Now redo clustering with the remaining towers, seeding with highest energy tower
    // fa la stessa cosa di prima ma questa volta cerca ovunque eventali cluster al di fuori della traiettoria
    nclu=0; //numero di cluster
    eclumax=0; //energia massima totale tra i cluster
    ncellmax=0; //numero massimo di celle tra i cluster
    
    int imaxx;
    int imaxy;
    int imaxz;
    
    avge =0;
    avgcells=0;
    
    do {
      
      //initialize
      for (int index=0; index<nmaxcells; index++) {
	vz[index]=0;
	vx[index]=0;
	vy[index]=0;
	ve[index]=0.;
      }
      ncell=0;
      eclu=0;
      double emax = emin;
      imaxx = -1;
      imaxy = -1;
      imaxz = -1;
      for (int iz=0; iz<50; iz++) {
	for (int ix=0; ix<32; ix++) {
	  for (int iy=0; iy<32; iy++) {
	    if (e3d[ix][iy][iz]>emax) {
	      imaxx=ix;
	      imaxy=iy;
	      imaxz=iz;
	      emax=e3d[ix][iy][iz];
	    }
	  }
	}
      }
      if (imaxx>-1) {
	nclu++;
	vz[ncell]=imaxz;
	vx[ncell]=imaxx;
	vy[ncell]=imaxy;
	ve[ncell]=e3d[imaxx][imaxy][imaxz];
	eclu+=ve[ncell];
	e3d[imaxx][imaxy][imaxz]=0.;
	ncell++;
	cluster3d(ncell);
      }
      avge += eclu;
      avgcells += ncell;
      if (eclu>eclumax) eclumax=eclu;
      if (ncell>ncellmax) ncellmax=ncell;
      //if (eclu < eclumin) eclumin = eclu;
      
      
    } while (imaxx>-1); 
    
    data[17] = nclu;
    data[18] = ncellmax;
    data[19] = eclumax;
    
    if(nclu == 0){
      data[20] = 0;
      data[21] = 0;
    }
    else{
      //data[22] = avge/avgcells;
      if (ncellmax != 0) data[20] = eclumax/ncellmax;//avge/avgcells;
      else data[20] =0;
      data[21] = avgcells/nclu;
    }
    
    // Search for 3x3 clusters not aligned with muon direction
    double maxsum=0.;
    int kmaxsum = -1;
    double prevmax=0;
    double rmaxsum = 0.;
    
    for (int i=0; i<30; i++) {
      for (int j=0; j<30; j++) {
	if (ixm-i>=0 && ixm-i<2 && iym-j>0 && iym-j<2) continue;
	for (int k=0; k<48; k++) {
	  double sum = 0;
	  for (int kk=0; kk<3; kk++) {
	    sum = sum + e3d[i][j][k+kk]   + e3d[i+1][j][k+kk]   + e3d[i+2][j][k+kk]+
	      e3d[i][j+1][k+kk] + e3d[i+1][j+1][k+kk] + e3d[i+2][j+1][k+kk]+
	      e3d[i][j+2][k+kk] + e3d[i+1][j+2][k+kk] + e3d[i+2][j+2][k+kk];
	  }
	  if (sum>maxsum) {
	    prevmax = maxsum;
	    maxsum = sum;
	    kmaxsum = k;
	    rmaxsum = sqrt((i-ixm)*(i-ixm)+(j-iym)*(j-iym));
	  }
	}
      }
    }
    
    data[22] = xmom;
    data[23] = ymom;
    data[24] = emeas;	
    data[25] = maxsum;
    
    // Override var(10) and var(15) from previous runs
    data[19] = prevmax;
    
    //-------------------------------------------------------------------------
    
    // massimo,minimo, sqm, mean di ogni feature
    for (int iv=0; iv<ND; iv++) {
      if (data[iv]<mina[iv]) mina[iv]=data[iv];
      if (data[iv]>maxa[iv]) maxa[iv]=data[iv];
    }
    for (int dat=0; dat<ND; dat++) { 
      results << data[dat] << " "; 
      // Calculate sqm of each feature
      avgdata[dat]+=data[dat];
      avgdata2[dat]+=data[dat]*data[dat];
    }
    for (int dat= 0; dat < 2; dat++){
      results << ebelow_[dat] << " "; 
    }
    navg++;
    results << true_energy*charge << endl;
    
  } // end loop on jentry
  
  
  
  cout << progress[51] << endl;
  cout << endl;
  
  results.close();
  
  // Dump average and sqm of each variable if wrote data to file
  cout << "Wrote to file " << navg << " events " << endl << endl;
  for (int i=0; i<ND; i++) {
    cout << "Var " << i << ": min= " << mina[i] << " max= " << maxa[i] << " avg = " << avgdata[i]/navg << ",  sqm = " << sqrt(avgdata2[i]/navg - pow(avgdata[i]/navg,2)) << endl;
  }
  
  
  // 	TCanvas * C = new TCanvas ("C","",900,500);
  // 	C->Divide(2,1);
  // 	C->cd(1);
  // 	Diff->Draw();
  // 	C->cd(2);
  // 	DiffE->SetMarkerStyle(20);
  // 	DiffE->Draw("COL4");
  
  // 	TCanvas * C2 = new TCanvas ("C2","",1000,600);
  // 	C2->Divide(5,2);
  // 	for (int ie=0; ie<10; ie++) {
  // 	C2->cd(ie+1);
  // 	DE[ie]->Draw();
  // 	}

}
