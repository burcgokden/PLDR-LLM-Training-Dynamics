// Reproducible ferromagnetic Potts equilibrium sampler. No external libraries.
// Hamiltonian H=-sum of equal-spin bonds, with right/down periodic bonds.
// Sampling uses a fixed number of Wolff cluster updates, not a state-dependent
// stopping time. A separate local heat-bath kernel supports independent checks.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 11) throw std::runtime_error("q L T seed chains samples burn thin kernel output");
    const int q=std::stoi(argv[1]), L=std::stoi(argv[2]), chains=std::stoi(argv[5]);
    const int samples=std::stoi(argv[6]), burn=std::stoi(argv[7]), thin=std::stoi(argv[8]);
    const double T=std::stod(argv[3]);
    const uint64_t seed=std::stoull(argv[4]);
    const std::string kernel=argv[9];
    if (q<2 || q>3 || L<2 || L>256 || !(T>0) || chains<1 || samples<1 || burn<0 || thin<1)
      throw std::runtime_error("invalid sampling design");
    if (kernel!="wolff" && kernel!="heatbath") throw std::runtime_error("unknown kernel");
    std::ifstream existing(argv[10],std::ios::binary);
    if(existing.good()) throw std::runtime_error("output already exists");
    std::ofstream output(argv[10],std::ios::binary);
    if(!output) throw std::runtime_error("cannot create output");
    const int V=L*L;
    std::vector<std::array<int,4>> neighbors(V);
    for(int i=0;i<V;++i) {
      int x=i/L,y=i%L;
      neighbors[i]={((x+1)%L)*L+y,((x+L-1)%L)*L+y,x*L+(y+1)%L,x*L+(y+L-1)%L};
    }
    const double bond=1-std::exp(-1/T);
    uint64_t changed=0, updates=0;
    for(int chain=0;chain<chains;++chain) {
      std::seed_seq ss{uint32_t(seed),uint32_t(seed>>32),uint32_t(chain),uint32_t(q),uint32_t(L)};
      std::mt19937_64 rng(ss);
      auto uniform=[&](){ return (rng()>>11)*0x1.0p-53; };
      auto integer=[&](int n){ return int(uniform()*n); };
      std::vector<uint8_t> s(V);
      // Half of the chains start ordered, half from an independent random state.
      for(auto &v:s) v=(chain%2==0)?0:integer(q);
      std::vector<int> cluster;cluster.reserve(V);
      auto step=[&](){
        if(kernel=="wolff") {
          int site=integer(V), old=s[site], next=(old+1+integer(q-1))%q;
          cluster.clear();cluster.push_back(site);s[site]=next;
          for(size_t j=0;j<cluster.size();++j)
            for(int k:neighbors[cluster[j]])
              if(s[k]==old && uniform()<bond) {s[k]=next;cluster.push_back(k);}
          changed+=cluster.size();
        } else {
          // V random-site heat-bath proposals form one sweep.
          for(int j=0;j<V;++j) {
            int site=integer(V);std::array<double,3> p{0,0,0};double total=0;
            for(int a=0;a<q;++a) {
              int count=0;for(int k:neighbors[site]) count+=(s[k]==a);
              total+=(p[a]=std::exp(count/T));
            }
            double u=uniform()*total;int a=0;while(a<q-1 && (u-=p[a])>0)++a;
            changed+=(s[site]!=a);s[site]=a;
          }
        }
        ++updates;
      };
      for(int j=0;j<burn;++j)step();
      for(int sample=0;sample<samples;++sample) {
        for(int j=0;j<thin;++j)step();
        output.write(reinterpret_cast<const char*>(s.data()),V);
      }
    }
    if(!output) throw std::runtime_error("write failed");
    std::cout<<"{\"updates\":"<<updates<<",\"changed_sites\":"<<changed
             <<",\"samples\":"<<uint64_t(chains)*samples<<"}\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
