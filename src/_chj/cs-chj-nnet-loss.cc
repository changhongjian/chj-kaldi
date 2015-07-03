// nnet/nnet-loss.cc

// Copyright 2011  Brno University of Technology (author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "cs-chj-nnet-loss.h"
#include "cudamatrix/cu-math.h"

//////////---------------
#include "_chj/chj.h"
#include "base/timer.h"
#include <unistd.h>  //有fork
#include <sstream>
#include <iterator>
#include <limits>
#include </usr/include/boost/math/special_functions/fpclassify.hpp>
#include "_chjcuda/chj-cu-kernels.h"

namespace chj{
namespace math{
using namespace kaldi;
BaseFloat log_add_mul(std::vector<BaseFloat> & vec,bool hasdone=false){
	int n=vec.size();
	KALDI_ASSERT(n>0);
	BaseFloat a=vec[n-1];
	vec.pop_back();
	if(!hasdone){
		KALDI_ASSERT(a>0);
		a=Log(a);
	}
	BaseFloat c=a;
	////
	for(std::vector<BaseFloat>::iterator it=vec.begin();it!=vec.end();it++){
		if(!hasdone){
            KALDI_ASSERT(*it>0);
            *it=Log(*it);
        }
		a=LogAdd(a,*it);
	}
/*	
 	if(a>0){
		for(int i=0;i<n-1;i++){
			chj_ss<<vec[i]<<" ";
		}
		chj_ss<<c<<"-----"<<endl;chj_pt();
	}
*/
	vec.clear();
	return a;
}

}//namespace chj::math
}//namespace chj

namespace kaldi {
namespace nnet1 {

void CTC::Run(const CuMatrix<BaseFloat> &nnet_out_, std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff_,int32 shrinkT){ //target目前就是一维,已经有blank

	Matrix<BaseFloat> nnet_out(nnet_out_);
    Matrix<BaseFloat> diff(nnet_out.NumRows(),nnet_out.NumCols(),kSetZero);

	const int32 BLANK=0;
	const BaseFloat NOT_LOG_VALUE=999;
	int32 num_pdf=nnet_out.NumCols(),pdf;
    int32 T=nnet_out.NumRows(),t;
    if(shrinkT>0) T=shrinkT;
	int32 S=target.size(),s; //sizeof(target)/sizeof(BaseFloat);
	KALDI_ASSERT(T>=2);
	Timer time;
	std::vector<BaseFloat> log_add_vec;
	Matrix<BaseFloat> alpha(T,S,kSetZero);
	Matrix<BaseFloat> beta(T,S,kSetZero);
	Matrix<BaseFloat> netout(nnet_out);
	//netout.Add(1e-39); 
	netout.ApplyLog(); //--- use log !
	//step 1: forward variable ----------------
	//
	t=0,s=0;
	alpha.Add(NOT_LOG_VALUE);//it should have been less or equal than 0
	if(nnet_out(t,target[s])>0){
		alpha(t,s)=netout(t,target[s]);
	}
	if(nnet_out(t,target[s+1])>0){
        alpha(t,s+1)=netout(t,target[s+1]);
    }
	for(t=1;t<T;t++){
		int32 tmp=S-2*(T-t);
		for(s=0;s<S;s++){
			if(s<tmp) continue;
			if(nnet_out(t,target[s])==0)continue;
			if(alpha(t-1,s)!=NOT_LOG_VALUE){
				log_add_vec.push_back(alpha(t-1,s));
			}
			if(s>=2){
				if(alpha(t-1,s-1)!=NOT_LOG_VALUE){
					log_add_vec.push_back(alpha(t-1,s-1));
				}
				if(target[s]!=BLANK && target[s]!=target[s-2]){
					if(alpha(t-1,s-2)!=NOT_LOG_VALUE){
						log_add_vec.push_back(alpha(t-1,s-2));
					}
				}
			}else if(s==1){//0的已经算在里面来了
				if(alpha(t-1,s-1)!=NOT_LOG_VALUE){
					log_add_vec.push_back(alpha(t-1,s-1));
				}
			}
			if(log_add_vec.size()>0){
				alpha(t,s)=netout(t,target[s])+chj::math::log_add_mul(log_add_vec,true);
			}
		}
	}
	//KALDI_LOG <<"22222 time elapsed = "<< time.Elapsed()/60 << " min; "<< std::endl ;
	//chj_ss<<"CTC-loss 22222 time elapsed = "<< time.Elapsed()/60 << " min; " <<endl;chj_pt();
	//step 2: backward variable ----------------
	t=T-1,s=S-1;
	beta.Add(NOT_LOG_VALUE);
	beta(t,s)=0; //pdf=0 it means blank,  it is log value log(1)
	beta(t,s-1)=0;
	BaseFloat beta_netout=0;
	for(--t;t>=0;t--){
		int32 tmp=2*t+1;
		for(s=S-1; s>=0;s--){
			if(s>tmp) continue;
			if(beta(t+1,s)!=NOT_LOG_VALUE && nnet_out(t+1,target[s])>0){
				beta_netout=beta(t+1,s)+netout(t+1,target[s]);
				log_add_vec.push_back( beta_netout );
			}
			if(s<S-2){
				if(beta(t+1,s+1)!=NOT_LOG_VALUE && nnet_out(t+1,target[s+1])>0){
					beta_netout=beta(t+1,s+1)+netout(t+1,target[s+1]);
	                log_add_vec.push_back( beta_netout );
				}
				if(target[s]!=BLANK && target[s]!=target[s+2]){
					if(beta(t+1,s+2)!=NOT_LOG_VALUE && nnet_out(t+1,target[s+2])>0){
						beta_netout=beta(t+1,s+2)+netout(t+1,target[s+2]);
			            log_add_vec.push_back( beta_netout );
					}
				}
			}else if(s==S-2){
				if(beta(t+1,s+1)!=NOT_LOG_VALUE && nnet_out(t+1,target[s+1])>0){
					beta_netout=beta(t+1,s+1)+netout(t+1,target[s+1]);
                    log_add_vec.push_back( beta_netout );
				}
			}
			if(log_add_vec.size()>0){
				beta(t,s)=chj::math::log_add_mul(log_add_vec,true);
			}
		}
	}
	//KALDI_LOG <<"CTC-loss 33333 time elapsed = "<< time.Elapsed()/60 << " min; "<< std::endl ;
	//chj_ss<<"CTC-loss 33333 time elapsed = "<< time.Elapsed()/60 << " min; " <<endl;chj_pt();
	//step 3: error signal ----------------
	diff=nnet_out;  // =号运算这里是复制，而不是引用
	//*****netout  以下具有不同的含义
	netout.Resize(T,S);
	netout.SetZero(); // 所有的都清空
	netout.Add(NOT_LOG_VALUE);
	for(t=0;t<T;t++){
		for(s=0;s<S;s++){
			if(alpha(t,s)!=NOT_LOG_VALUE && beta(t,s)!=NOT_LOG_VALUE){
				netout(t,s)=alpha(t,s)+beta(t,s);
			}
		}
	}

// 这里就是按照 p 一样来处理的
	BaseFloat p=NOT_LOG_VALUE;
	t=0;
	for(s=0;s<S;s++){
        if(netout(t,s)!=NOT_LOG_VALUE){
            log_add_vec.push_back( netout(t,s) );
        }
    }
//    if(log_add_vec.size()>0){
        p=chj::math::log_add_mul(log_add_vec,true); //按道理p是一样的
        KALDI_ASSERT(!isinf(p));
        KALDI_ASSERT(p==p);
        KALDI_ASSERT(p!=NOT_LOG_VALUE);
//	}
 //   std::cerr<<p<<"----------------------"<<endl;

	loss_-=T*p;


	//KALDI_LOG <<"CTC-loss 44444 time elapsed = "<< time.Elapsed()/60 << " min; "<< std::endl ;
	//chj_ss<<"CTC-loss 44444 time elapsed = "<< time.Elapsed()/60 << " min; " <<endl;chj_pt();
	std::vector<int32> *lab=new std::vector<int32>[num_pdf];
	for(s=0;s<S;s++) lab[target[s]].push_back(s);
	for(t=0;t<T;t++){
/*
		BaseFloat p=NOT_LOG_VALUE;
		for(s=0;s<S;s++){
			if(netout(t,s)!=NOT_LOG_VALUE){
				log_add_vec.push_back( netout(t,s) );
			}
		}
		if(log_add_vec.size()>0){
			p=chj::math::log_add_mul(log_add_vec,true); //按道理p是一样的
            KALDI_ASSERT(!isinf(p));
            KALDI_ASSERT(p==p);
            KALDI_ASSERT(p!=NOT_LOG_VALUE);
		}

		loss_-=p; //***
*/
		for(pdf=0;pdf<num_pdf;pdf++){
			BaseFloat tmp=0;
			std::vector<int32> labk=lab[pdf];
			int32 n=labk.size();
			for(int i=0;i<n;i++){
				if(netout(t,labk[i])!=NOT_LOG_VALUE){
					log_add_vec.push_back( netout(t,labk[i]) );
				}
			}
			
			if(log_add_vec.size()>0){
				tmp=chj::math::log_add_mul(log_add_vec,true);
				//chj_ss<<"diff: "<<diff(t,pdf)<<"   "<<tmp<<"   "<<p<<"  "<<endl;chj_pt();
				(diff)(t,pdf)=(diff)(t,pdf)-Exp(tmp-p); //7.29
				
			}
		}
	}
	delete [] lab;
	//diff.
	*diff_=diff;
	frames_+=T;
	sentences_+=1;
	//chj_ss<<"sent: -> "<<sentences_<<endl;chj_pt();
	//KALDI_LOG <<"CTC-loss time elapsed = "<< time.Elapsed()/60 << " min; "<< std::endl ;
	//chj_ss<<"CTC-loss time elapsed = "<< time.Elapsed()/60 << " min; " <<endl;chj_pt();
	//
	std::cerr<<time.Elapsed()<<endl;
	
} //end Run

void CTC::RunSpeedUp(const CuMatrix<BaseFloat> &nnet_out_, std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff_,int32 shrinkT){
    Matrix<BaseFloat> nnet_out(nnet_out_);
    //Matrix<BaseFloat> diff(nnet_out.NumRows(),nnet_out.NumCols(),kSetZero);   这个里面完全不用diff
	const int32 BLANK=0;
	const BaseFloat NOT_LOG_VALUE=999;
	int32 num_pdf=nnet_out.NumCols(),pdf;
    int32 T=nnet_out.NumRows(),t;
    if(shrinkT>0) T=shrinkT;
	int32 S=target.size(),s; //sizeof(target)/sizeof(BaseFloat);
	KALDI_ASSERT(T>=2);
	Timer time;
	std::vector<BaseFloat> log_add_vec;
	Matrix<BaseFloat> alpha(T,S,kSetZero);
	Matrix<BaseFloat> beta(T,S,kSetZero);
	Matrix<BaseFloat> netout(nnet_out);
	//netout.Add(1e-39); 
	netout.ApplyLog(); //--- use log !
	//step 1: forward variable ----------------
	//
	t=0,s=0;
	alpha.Add(NOT_LOG_VALUE);//it should have been less or equal than 0
	if(nnet_out(t,target[s])>0){
		alpha(t,s)=netout(t,target[s]);
	}
	if(nnet_out(t,target[s+1])>0){
        alpha(t,s+1)=netout(t,target[s+1]);
    }
	for(t=1;t<T;t++){
		int32 tmp=S-2*(T-t);
		for(s=0;s<S;s++){
			if(s<tmp) continue;
			if(nnet_out(t,target[s])==0)continue;
			if(alpha(t-1,s)!=NOT_LOG_VALUE){
				log_add_vec.push_back(alpha(t-1,s));
			}
			if(s>=2){
				if(alpha(t-1,s-1)!=NOT_LOG_VALUE){
					log_add_vec.push_back(alpha(t-1,s-1));
				}
				if(target[s]!=BLANK && target[s]!=target[s-2]){
					if(alpha(t-1,s-2)!=NOT_LOG_VALUE){
						log_add_vec.push_back(alpha(t-1,s-2));
					}
				}
			}else if(s==1){//0的已经算在里面来了
				if(alpha(t-1,s-1)!=NOT_LOG_VALUE){
					log_add_vec.push_back(alpha(t-1,s-1));
				}
			}
			if(log_add_vec.size()>0){
				alpha(t,s)=netout(t,target[s])+chj::math::log_add_mul(log_add_vec,true);
			}
		}
	}
	//KALDI_LOG <<"22222 time elapsed = "<< time.Elapsed()/60 << " min; "<< std::endl ;
	//chj_ss<<"CTC-loss 22222 time elapsed = "<< time.Elapsed()/60 << " min; " <<endl;chj_pt();
	//step 2: backward variable ----------------
	t=T-1,s=S-1;
	beta.Add(NOT_LOG_VALUE);
	beta(t,s)=0; //pdf=0 it means blank,  it is log value log(1)
	beta(t,s-1)=0;
	BaseFloat beta_netout=0;
	for(--t;t>=0;t--){
		int32 tmp=2*t+1;
		for(s=S-1; s>=0;s--){
			if(s>tmp) continue;
			if(beta(t+1,s)!=NOT_LOG_VALUE && nnet_out(t+1,target[s])>0){
				beta_netout=beta(t+1,s)+netout(t+1,target[s]);
				log_add_vec.push_back( beta_netout );
			}
			if(s<S-2){
				if(beta(t+1,s+1)!=NOT_LOG_VALUE && nnet_out(t+1,target[s+1])>0){
					beta_netout=beta(t+1,s+1)+netout(t+1,target[s+1]);
	                log_add_vec.push_back( beta_netout );
				}
				if(target[s]!=BLANK && target[s]!=target[s+2]){
					if(beta(t+1,s+2)!=NOT_LOG_VALUE && nnet_out(t+1,target[s+2])>0){
						beta_netout=beta(t+1,s+2)+netout(t+1,target[s+2]);
			            log_add_vec.push_back( beta_netout );
					}
				}
			}else if(s==S-2){
				if(beta(t+1,s+1)!=NOT_LOG_VALUE && nnet_out(t+1,target[s+1])>0){
					beta_netout=beta(t+1,s+1)+netout(t+1,target[s+1]);
                    log_add_vec.push_back( beta_netout );
				}
			}
			if(log_add_vec.size()>0){
				beta(t,s)=chj::math::log_add_mul(log_add_vec,true);
			}
		}
	}
    
    
    	//KALDI_LOG <<"CTC-loss 33333 time elapsed = "<< time.Elapsed()/60 << " min; "<< std::endl ;
	//chj_ss<<"CTC-loss 33333 time elapsed = "<< time.Elapsed()/60 << " min; " <<endl;chj_pt();
	//step 3: error signal ----------------
	//*****netout  以下具有不同的含义
	netout.Resize(T,S);
	netout.SetZero(); // 所有的都清空
	//netout.Add(NOT_LOG_VALUE);
	/*for(t=0;t<T;t++){
		for(s=0;s<S;s++){
			if(alpha(t,s)!=NOT_LOG_VALUE && beta(t,s)!=NOT_LOG_VALUE){
				netout(t,s)=alpha(t,s)+beta(t,s);
			}
		}
	}*/
    
/*
    CuMatrix<BaseFloat> cu_alpha_beta(netout);
	CuMatrix<BaseFloat> cu_alpha(alpha);
	CuMatrix<BaseFloat> cu_beta(beta);

    chj::cuda::mat_add_mat(cu_alpha_beta.Dim(),cu_alpha_beta.CHJ_Data(),
        cu_alpha.CHJ_Data(),cu_beta.CHJ_Data(),NOT_LOG_VALUE);

    netout=Matrix<BaseFloat>(cu_alpha_beta);
    alpha.Resize(0,0); beta.Resize(0,0); // free
*/
    for(t=0;t<T;t++){
        for(s=0;s<S;s++){
            if(alpha(t,s)!=NOT_LOG_VALUE && beta(t,s)!=NOT_LOG_VALUE){
                netout(t,s)=alpha(t,s)+beta(t,s);
            }
        }
    }
    alpha.Resize(0,0); beta.Resize(0,0);
    CuMatrix<BaseFloat> cu_alpha_beta(netout);
 
	//KALDI_LOG <<"CTC-loss 44444 time elapsed = "<< time.Elapsed()/60 << " min; "<< std::endl ;
	//chj_ss<<"CTC-loss 44444 time elapsed = "<< time.Elapsed()/60 << " min; " <<endl;chj_pt();
    
    // important ,下面我假设 p 的值不会改变 
    BaseFloat p=NOT_LOG_VALUE;
    for(s=0;s<S;s++){
        if(netout(0,s)!=NOT_LOG_VALUE){
            log_add_vec.push_back( netout(0,s) );
        }
    }
 //   if(log_add_vec.size()>0){
        p=chj::math::log_add_mul(log_add_vec,true); //按道理p是一样的
        KALDI_ASSERT(!isinf(p));
        KALDI_ASSERT(p==p);
//if(p>0){
	
//   std::cerr<<p<<"-----------------"<<endl;
//}
    //    KALDI_ASSERT(p<=0);
 //   }
    loss_-=T*p; //***
    std::vector<int32> *lab=new std::vector<int32>[num_pdf];
	for(s=0;s<S;s++) lab[target[s]].push_back(s);
    int32 lablenmax=0;
    for(pdf=0;pdf<num_pdf;pdf++){
        if(lablenmax<lab[pdf].size()) lablenmax=lab[pdf].size();
    }
    
    Matrix<BaseFloat>host_lab(num_pdf,lablenmax+1,kSetZero);
    for(pdf=0;pdf<num_pdf;pdf++){
        int n=lab[pdf].size();
        host_lab(pdf,0)=n;
        for(int i=1;i<=n;i++){
            host_lab(pdf,i)=lab[pdf][i-1];
        }
    }
    
    
	/*for(t=0;t<T;t++){
		for(pdf=0;pdf<num_pdf;pdf++){
			BaseFloat tmp=0;
			std::vector<int32> labk=lab[pdf];
			int32 n=labk.size();
			for(int i=0;i<n;i++){
				if(netout(t,labk[i])!=NOT_LOG_VALUE){
					log_add_vec.push_back( netout(t,labk[i]) );
				}
			}
			if(log_add_vec.size()>0){
				tmp=chj::math::log_add_mul(log_add_vec,true);
				//chj_ss<<"diff: "<<diff(t,pdf)<<"   "<<tmp<<"   "<<p<<"  "<<endl;chj_pt();
				(diff)(t,pdf)=(diff)(t,pdf)-Exp(tmp-p); //7.29			
			}
		}
	}*/
//	CuMatrix<BaseFloat> cutmp= CuMatrix<BaseFloat>(nnet_out); 
//	*diff_=cutmp;
	diff_->Resize(nnet_out.NumRows(),nnet_out.NumCols());

	diff_->CopyFromMat(nnet_out);

//    *diff_= CuMatrix<BaseFloat>(nnet_out); //用new  会造成内存泄漏
	CuMatrix<BaseFloat> cu_lab=CuMatrix<BaseFloat>(host_lab);
 

    chj::cuda::ctc_loss_fun(diff_->Dim(),diff_->CHJ_Data(),
        cu_alpha_beta.Dim(),cu_alpha_beta.CHJ_Data(),cu_lab.Dim(),cu_lab.CHJ_Data(),p,NOT_LOG_VALUE); //函数就不提供了

	delete [] lab;
	//diff.
	frames_+=T;
	sentences_+=1;
	//KALDI_LOG <<"CTC-loss time elapsed = "<< time.Elapsed()/60 << " min; "<< std::endl ;
	//chj_ss<<"CTC-loss time elapsed = "<< time.Elapsed()/60 << " min; " <<endl;chj_pt();
	std::cerr<<time.Elapsed()<<endl;
    
} //end RunSpeedUp

std::string CTC::Report() {
	std::ostringstream oss;
	oss << "AvgLoss: " << loss_/frames_ << " (CTC), "<< std::endl;
	oss << "FRAME_ACCURACY >> " << "0" << "% <<"<< std::endl;; //仅为了统一格式
	oss << "Sentence "<<sentences_<<std::endl;
	return oss.str(); 
}

} // namespace nnet1
} // namespace kaldi
