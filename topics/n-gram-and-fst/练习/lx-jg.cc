#include "lm/model.hh"
#include <iostream>
#include <string>
#include <sstream>
using namespace std;
string num2str(int i)
{
    stringstream ss;
    ss<<i;
    return ss.str();
}

int main(int argn,char *argv[]) {
  using namespace lm::ngram;
  Model model(argv[1]);
  State state(model.BeginSentenceState()), out_state;
  lm::FullScoreReturn ret;
  const Vocabulary &vocab = model.GetVocabulary();
  unsigned int history[6]={0};
  while (std::cin >> history[2] >> history[1] >> history[0] >> history[3]  ) {
    history[0]=vocab.Index( num2str( history[0] ) );	
	history[1]=vocab.Index( num2str( history[1] ) );
	history[2]=vocab.Index( num2str( history[2] ) );
    history[3]=vocab.Index( num2str( history[3] ) );	
	ret=model.FullScoreForgotState(history, history+3, history[3], out_state);
    std::cout << ret.prob<< "    " << ret.prob * 2.302585  << '\n';
  }

/*
  while (std::cin >> history[1] >> history[0] >> history[2]  ) {
    history[0]=vocab.Index( num2str( history[0] ) );	
	history[1]=vocab.Index( num2str( history[1] ) );
	history[2]=vocab.Index( num2str( history[2] ) );
	ret=model.FullScoreForgotState(history, history+2, history[2], out_state);
    std::cout << ret.prob << '\n';
  }
*/
 return 0;
}
