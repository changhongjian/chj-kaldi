#ifndef CHJ_H_
#define CHJ_H_
#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
using namespace std;
//namespace kaldi {
    static string chj_logname="chj.log";
    static stringstream chj_ss;//
    static void chj_pt(){
       FILE * fp=fopen(chj_logname.c_str(),"a"); 
       fputs(chj_ss.str().c_str(),fp);
       fclose(fp);
       chj_ss.str("");// *** the true method to clear ss
       chj_ss.clear();
    }
    static void chj_setfile(){
        FILE * fp=fopen(chj_logname.c_str(),"w"); 
        fclose(fp);
    }
//}
#endif 
