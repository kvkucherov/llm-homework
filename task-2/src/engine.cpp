// Original Battlesnake search. No code from other competitors is used.
// The transition ordering follows BattlesnakeOfficial/rules (see README).
#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr int CELLS = 625, INF = 10000000, WIN = 1000000;
// Standard 11x11 boards fit in two machine words. Larger boards use the queue
// implementation below; no bit shift is performed for a cell outside 0..127.
using Bits=__uint128_t;
using Clock = std::chrono::steady_clock;
struct Snake { std::vector<int> body; int health = 100; bool alive = true; };
struct State { std::vector<Snake> snakes; std::bitset<CELLS> food; int ply=0; };
struct Config {
    int w=11, h=11, damage=14; bool wrapped=false, constrictor=false;
    std::array<int,CELLS> hazard{};
    std::array<std::array<int,4>,CELLS> next{};
    Bits boardMask=0,leftEdge=0,rightEdge=0,topEdge=0,bottomEdge=0;
    Bits expand(Bits b) const {
        Bits out=(b<<w)|(b>>w)|((b&~leftEdge)>>1)|((b&~rightEdge)<<1);
        if(wrapped)out|=((b&leftEdge)<<(w-1))|((b&rightEdge)>>(w-1))|
                        ((b&topEdge)>>(w*(h-1)))|((b&bottomEdge)<<(w*(h-1)));
        return out&boardMask;
    }
    void init() {
        if(w*h<=128) {
            boardMask=~Bits(0)>>(128-w*h);
            for(int y=0;y<h;y++){leftEdge|=Bits(1)<<(y*w);rightEdge|=Bits(1)<<(y*w+w-1);}
            bottomEdge=(Bits(1)<<w)-1;topEdge=bottomEdge<<(w*(h-1));
        }
        for(int p=0;p<w*h;p++) for(int d=0;d<4;d++) {
            int x=p%w+(d==3)-(d==2), y=p/w+(d==0)-(d==1);
            if(wrapped) {x=(x+w)%w; y=(y+h)%h;}
            next[p][d]=(x<0||y<0||x>=w||y>=h) ? -1 : y*w+x;
        }
    }
};

State decode(const int* data,int size,Config& c) {
    int k=0;
    auto read=[&](){if(k>=size)throw std::runtime_error("short packet");return data[k++];};
    c.w=read();c.h=read();c.damage=read();c.wrapped=read()!=0;c.constrictor=read()!=0;
    if(c.w<2||c.h<2||c.w>25||c.h>25||c.damage<0||c.damage>1000)throw std::runtime_error("bad board");
    auto pos=[&](){int p=read();if(p<0||p>=c.w*c.h)throw std::runtime_error("bad position");return p;};
    State s;
    int n=read();if(n<0||n>CELLS)throw std::runtime_error("food count");
    for(int i=0;i<n;i++)s.food.set(pos());
    n=read();if(n<0||n>CELLS*16)throw std::runtime_error("hazard count");
    for(int i=0;i<n;i++)c.hazard[pos()]++;
    n=read();if(n<1||n>16)throw std::runtime_error("snake count");
    for(int i=0;i<n;i++) {
        Snake a;int len=read();a.health=read();
        if(len<1||len>CELLS+128||a.health<0||a.health>100)throw std::runtime_error("snake data");
        a.alive=a.health>0;for(int j=0;j<len;j++)a.body.push_back(pos());s.snakes.push_back(std::move(a));
    }
    c.init();return s;
}

State step(const State& s,const std::vector<int>& moves,const Config& c) {
    State t=s;t.ply++;
    for(size_t i=0;i<t.snakes.size();i++) {
        auto& a=t.snakes[i];if(!a.alive)continue;
        int head=c.next[a.body[0]][moves[i]];
        // Standard rules move the tail even on eating, then duplicate the NEW tail.
        for(int j=int(a.body.size())-1;j>0;j--)a.body[j]=a.body[j-1];
        a.body[0]=head;a.health--;
        bool eat=head>=0&&s.food[head];
        if(head>=0&&!eat)a.health-=c.damage*c.hazard[head];
        if(eat||c.constrictor) {a.body.push_back(a.body.back());a.health=100;}
        if(eat)t.food.reset(head);
        if(head<0||a.health<=0)a.alive=false;
    }
    std::array<int,CELLS> occupied{};
    for(const auto& a:t.snakes)if(a.alive)
        for(size_t j=1;j<a.body.size();j++)if(a.body[j]>=0)occupied[a.body[j]]++;
    std::vector<bool> killed(t.snakes.size(),false);
    for(size_t i=0;i<t.snakes.size();i++) {
        const auto& a=t.snakes[i];if(!a.alive)continue;
        if(occupied[a.body[0]]) {killed[i]=true;continue;}
        for(size_t j=0;j<t.snakes.size();j++) {
            const auto& b=t.snakes[j];
            if(i!=j&&b.alive&&a.body[0]==b.body[0]&&a.body.size()<=b.body.size())killed[i]=true;
        }
    }
    for(size_t i=0;i<t.snakes.size();i++)if(killed[i])t.snakes[i].alive=false;
    return t;
}

std::vector<int> legal(const State& s,int who,const Config& c) {
    if(!s.snakes[who].alive)return {0};
    std::array<bool,CELLS> blocked{};
    for(const auto& a:s.snakes)if(a.alive) {
        int end=int(a.body.size())-(c.constrictor?0:1);
        // Constrictor also duplicates the new tail, so the old tail vacates;
        // the old penultimate remains. For a grown snake they are already equal.
        if(c.constrictor)end=std::max(0,int(a.body.size())-1);
        for(int j=0;j<end;j++)blocked[a.body[j]]=true;
    }
    std::vector<int> out;
    const auto& a=s.snakes[who];
    for(int d=0;d<4;d++) {
        int p=c.next[a.body[0]][d];
        if(p<0||blocked[p])continue;
        if(!c.constrictor&&!s.food[p]&&a.health<=1+c.damage*c.hazard[p])continue;
        out.push_back(d);
    }
    // Keep a move even in forced-death states so simultaneous draws are evaluated.
    if(out.empty())out.push_back(0);
    return out;
}

struct Reach {
    std::array<int,CELLS> dist{};
    int area=0, food=1000, staticArea=0;
};
Reach reach(const State& s,int who,const Config& c,const std::array<int,CELLS>& release) {
    Reach r;r.dist.fill(10000);
    const auto& a=s.snakes[who];if(!a.alive)return r;
    std::array<int,CELLS> q{};int begin=0,end=0;
    int head=a.body[0];q[end++]=head;r.dist[head]=0;
    while(begin<end) {
        int p=q[begin++];r.area++;
        if(s.food[p])r.food=std::min(r.food,r.dist[p]);
        for(int d=0;d<4;d++) {
            int v=c.next[p][d], nd=r.dist[p]+1;
            if(v<0||r.dist[v]!=10000||release[v]>nd)continue;
            r.dist[v]=nd;q[end++]=v;
        }
    }
    // Immediate empty-space flood guards against optimistic tail release paths.
    std::array<bool,CELLS> seen{};begin=0;end=0;q[end++]=head;seen[head]=true;
    while(begin<end) {
        int p=q[begin++];r.staticArea++;
        for(int d=0;d<4;d++) {int v=c.next[p][d];if(v<0||seen[v]||release[v]>1)continue;seen[v]=true;q[end++]=v;}
    }
    return r;
}

struct Weights {
    double length=65, territory=3.0, food=42, space=16, aggression=1.0;
};

struct BitFlood {
    // Cells become available in the same BFS layer in which a tail vacates.
    std::array<Bits,128> freed{};
    Bits open=0,food=0;
    BitFlood()=default;
    BitFlood(const State& s,const Config& c,const std::array<int,CELLS>& release) {
        init(s,c,release);
    }
    void init(const State& s,const Config& c,const std::array<int,CELLS>& release) {
        for(int p=0;p<c.w*c.h;p++) {
            Bits cell=Bits(1)<<p;
            if(release[p]<=1)open|=cell;
            else if(release[p]<128)freed[release[p]]|=cell;
            if(s.food[p])food|=cell;
        }
    }
};
int population(Bits b) {return __builtin_popcountll(uint64_t(b))+__builtin_popcountll(uint64_t(b>>64));}
int safeBitArea(const State& s,int who,const Config& c,const BitFlood& field,const std::vector<Reach>& rr) {
    // Enemy arrival times are an estimate, not a proof that a room is safe.
    std::array<Bits,128> dangerous{};
    for(int p=0;p<c.w*c.h;p++) {
        int arrival=10000;
        for(size_t j=0;j<s.snakes.size();j++)if(int(j)!=who&&s.snakes[j].alive)
            arrival=std::min(arrival,rr[j].dist[p]+int(s.snakes[j].body.size()<s.snakes[who].body.size()));
        if(arrival<128)dangerous[arrival]|=Bits(1)<<p;
    }
    Bits front=Bits(1)<<s.snakes[who].body[0],seen=front,open=field.open,unsafe=dangerous[0];
    for(int depth=0;front;depth++) {
        if(depth+1<128){open|=field.freed[depth+1];unsafe|=dangerous[depth+1];}
        front=c.expand(front)&open&~unsafe&~seen;seen|=front;
    }
    return population(seen);
}
Reach bitReach(const State& s,int who,const Config& c,const BitFlood& field) {
    Reach r;r.dist.fill(10000);
    const auto& a=s.snakes[who];if(!a.alive)return r;
    Bits front=Bits(1)<<a.body[0],seen=front,open=field.open;
    for(int depth=0;front;depth++) {
        if(front&field.food)r.food=std::min(r.food,depth);
        uint64_t lo=uint64_t(front),hi=uint64_t(front>>64);
        while(lo){int p=__builtin_ctzll(lo);r.dist[p]=depth;lo&=lo-1;}
        while(hi){int p=__builtin_ctzll(hi)+64;r.dist[p]=depth;hi&=hi-1;}
        if(depth+1<128)open|=field.freed[depth+1];
        front=c.expand(front)&open&~seen;seen|=front;
    }
    r.area=population(seen);
    front=Bits(1)<<a.body[0];seen=front;
    while(front){front=c.expand(front)&field.open&~seen;seen|=front;}
    r.staticArea=population(seen);return r;
}

double evaluate(const State& s,const Config& c,const Weights& w) {
    const auto& us=s.snakes[0];int live=0;
    for(size_t i=1;i<s.snakes.size();i++)live+=s.snakes[i].alive;
    if(!us.alive)return live==0 ? -15000 : -WIN+s.ply*50;
    if(live==0&&s.snakes.size()>1)return WIN-s.ply*50;
    std::array<int,CELLS> release{};
    for(const auto& a:s.snakes)if(a.alive)
        for(int j=0;j<int(a.body.size());j++)
            release[a.body[j]]=std::max(release[a.body[j]],c.constrictor?10000:int(a.body.size())-j);
    std::vector<Reach> rr;rr.reserve(s.snakes.size());
    BitFlood field;
    if(c.w*c.h<=128) {
        field.init(s,c,release);
        for(int i=0;i<int(s.snakes.size());i++)rr.push_back(bitReach(s,i,c,field));
    } else for(int i=0;i<int(s.snakes.size());i++)rr.push_back(reach(s,i,c,release));
    auto individual=[&](int i) {
        const auto& a=s.snakes[i];const auto& r=rr[i];int len=int(a.body.size());
        double value=0;
        int need=std::min(c.w*c.h,len+4);
        if(r.area<need)value-=w.space*std::pow(double(need-r.area),1.4);
        if(r.staticArea<len)value-=w.space*0.35*(len-r.staticArea);
        value+=0.35*std::min(r.area,len*2+8);
        if(live) {
            // Count connected escape space without crossing a cell an enemy can
            // take first. Ordinary flood-fill may count a whole room beyond an
            // opponent-controlled doorway, even when we cannot enter that room.
            int safeArea=0;
            if(c.w*c.h<=128) safeArea=safeBitArea(s,i,c,field,rr);
            else {
                std::array<int,CELLS> distance,queue;
                std::fill_n(distance.begin(),c.w*c.h,10000);
                int begin=0,end=1;queue[0]=a.body[0];distance[queue[0]]=0;
                while(begin<end) {
                    int p=queue[begin++],nd=distance[p]+1;
                    for(int d=0;d<4;d++) {
                        int v=c.next[p][d];
                        if(v<0||distance[v]!=10000||release[v]>nd)continue;
                        bool danger=false;
                        for(size_t j=0;j<s.snakes.size();j++)if(int(j)!=i&&s.snakes[j].alive) {
                            int rival=rr[j].dist[v];
                            if(rival<nd||(rival==nd&&s.snakes[j].body.size()>=a.body.size())){danger=true;break;}
                        }
                        if(!danger){distance[v]=nd;queue[end++]=v;}
                    }
                }
                safeArea=end;
            }
            int shortage=std::max(0,std::min(c.w*c.h,len+2)-safeArea);
            value-=w.space*std::pow(double(shortage),1.25);
        }
        if(!c.constrictor) {
            value+=0.35*a.health;
            if(r.food<1000) {
                value+=w.food/(1.0+0.35*r.food);
                int margin=a.health-r.food;
                if(margin<16)value-=8.0*(16-margin);
                if(margin<=0)value-=4000;
            } else if(a.health<25) value-=(25-a.health)*20;
        }
        int x=a.body[0]%c.w,y=a.body[0]/c.w;
        value-=0.7*(std::abs(x-(c.w-1)/2.0)+std::abs(y-(c.h-1)/2.0));
        value-=c.hazard[a.body[0]]*c.damage*2.0;
        return value;
    };
    double score=individual(0);double enemy=-INF;int maxLen=0;
    for(size_t i=1;i<s.snakes.size();i++)if(s.snakes[i].alive) {
        enemy=std::max(enemy,individual(int(i)));maxLen=std::max(maxLen,int(s.snakes[i].body.size()));
    }
    if(live)score-=enemy;
    int diff=int(us.body.size())-maxLen;
    if(!c.constrictor&&live) {
        score+=w.length*(std::clamp(diff,-4,4)+0.1*diff);
        if(diff>0)score+=w.length*0.8;
        if(diff<0)score-=w.length*0.8;
    }
    double territory=0;
    for(int p=0;p<c.w*c.h;p++) {
        int own=rr[0].dist[p];if(own>=10000)continue;
        bool wins=true,tie=false;
        for(size_t i=1;i<s.snakes.size();i++)if(s.snakes[i].alive) {
            int other=rr[i].dist[p];
            if(other<own||(other==own&&s.snakes[i].body.size()>us.body.size())){wins=false;break;}
            if(other==own&&s.snakes[i].body.size()==us.body.size())tie=true;
        }
        double cell=c.hazard[p]?0.3:1.0;
        if(wins)territory+=cell*(tie?0.25:1.0);
        else territory-=cell;
    }
    score+=w.territory*territory;
    // Length advantage is useful when we can occupy the opponent's exit space.
    if(diff>0&&live&&rr[0].area>int(us.body.size())+8) {
        int near=1000;
        for(size_t i=1;i<s.snakes.size();i++)if(s.snakes[i].alive)near=std::min(near,rr[0].dist[s.snakes[i].body[0]]);
        if(near<1000)score+=w.aggression*12.0/(1+near);
    }
    return score;
}

struct TTEntry {
    uint64_t key=0;double value=0;int depth=-1,bound=0,move=0;
    std::array<int,4> replies{{-1,-1,-1,-1}};
};
struct Timeout {};
struct Search {
    const Config& c;Weights weights;Clock::time_point deadline;
    uint64_t nodes=0;std::vector<TTEntry> tt;int best=0,completed=0;double value=0;
    Search(const Config& cfg,int ms,const double* w):c(cfg),deadline(Clock::now()+std::chrono::milliseconds(ms)),tt(1<<17) {
        if(w){weights.length=w[0];weights.territory=w[1];weights.food=w[2];weights.space=w[3];weights.aggression=w[4];}
    }
    void tick(){++nodes;if((nodes&63)==0&&Clock::now()>=deadline)throw Timeout{};}
    uint64_t hash(const State& s) {
        uint64_t h=1469598103934665603ULL;
        auto mix=[&](uint64_t v){h^=v+1;h*=1099511628211ULL;};
        for(const auto& a:s.snakes) {
            mix(a.alive);if(!a.alive)continue;mix(a.health);mix(a.body.size());for(int p:a.body)mix(p);
        }
        for(int p=0;p<c.w*c.h;p++)if(s.food[p])mix(p+1024);
        mix(s.ply);return h;
    }
    double search(const State& s,int depth,double alpha,double beta,int* root=nullptr) {
        tick();int live=0;for(size_t i=1;i<s.snakes.size();i++)live+=s.snakes[i].alive;
        if(!s.snakes[0].alive||(!live&&s.snakes.size()>1)||depth==0)return evaluate(s,c,weights);
        uint64_t key=hash(s);TTEntry cached=tt[key&(tt.size()-1)];
        double originalAlpha=alpha,originalBeta=beta;
        int pv=-1;
        if(cached.key==key) {
            pv=cached.move;
            if(cached.depth>=depth&&!root) {
                if(cached.bound==0)return cached.value;
                if(cached.bound==1)alpha=std::max(alpha,cached.value);else beta=std::min(beta,cached.value);
                if(alpha>=beta)return cached.value;
            }
        }
        auto ours=legal(s,0,c);
        std::vector<std::vector<int>> theirs(s.snakes.size());
        for(size_t i=1;i<s.snakes.size();i++)theirs[i]=legal(s,int(i),c);
        std::vector<int> actions(s.snakes.size(),0);
        // Cheap move ordering; alpha-beta and the previous iteration do the rest.
        auto quick=[&](int d){int p=c.next[s.snakes[0].body[0]][d];if(p<0)return -1e8;return (d==pv?1e7:0)+ (s.food[p]?100.0:0.0)-c.hazard[p]*20.0-std::abs(p%c.w-c.w/2)-std::abs(p/c.w-c.h/2);};
        std::stable_sort(ours.begin(),ours.end(),[&](int a,int b){return quick(a)>quick(b);});
        double bestValue=-INF;int bestMove=ours[0];
        std::array<int,4> replies{{-1,-1,-1,-1}};
        for(int m:ours) {
            actions[0]=m;double worst=INF;
            if(s.snakes.size()==2) {
                int response=cached.key==key?cached.replies[m]:-1;
                auto priority=[&](int d) {
                    int p=c.next[s.snakes[1].body[0]][d];
                    if(d==response)return 10000;
                    if(p<0)return -10000;
                    return (p==c.next[s.snakes[0].body[0]][m]?100:0)+(s.food[p]?10:0);
                };
                std::stable_sort(theirs[1].begin(),theirs[1].end(),[&](int a,int b){return priority(a)>priority(b);});
            }
            // Enumerate simultaneous opponent responses. In FFA this is paranoid
            // search; depth is limited by the same deadline, never by threads.
            auto enumerate=[&](auto&& self,size_t i)->void {
                if(worst<=alpha)return;
                if(i==s.snakes.size()) {
                    auto next=step(s,actions,c);
                    double result=search(next,depth-1,alpha,std::min(beta,worst));
                    if(result<worst) {
                        worst=result;
                        if(s.snakes.size()==2)replies[m]=actions[1];
                    }
                    return;
                }
                for(int move:theirs[i]){actions[i]=move;self(self,i+1);if(worst<=alpha)break;}
            };
            enumerate(enumerate,1);
            if(worst>bestValue){bestValue=worst;bestMove=m;}
            alpha=std::max(alpha,bestValue);if(alpha>=beta)break;
        }
        auto& entry=tt[key&(tt.size()-1)];
        entry={key,bestValue,depth,bestValue<=originalAlpha?2:(bestValue>=originalBeta?1:0),bestMove};
        entry.replies=replies;
        if(root)*root=bestMove;
        return bestValue;
    }
    int run(const State& s,int maxDepth) {
        best=legal(s,0,c)[0];
        for(int d=1;d<=maxDepth;d++) {
            int candidate=best;
            try {
                // Most iterations stay near the previous score. A narrow window
                // prunes sooner; a failed bound is widened until the result is exact.
                double v,window=d>2?60:INF;
                for(;;) {
                    double lo=std::max(double(-INF),value-window),hi=std::min(double(INF),value+window);
                    v=search(s,d,lo,hi,&candidate);
                    if((v>lo&&v<hi)||window>=2*INF)break;
                    window=std::min(double(2*INF),window*4);
                }
                best=candidate;value=v;completed=d;
            }
            catch(const Timeout&){break;}
            if(std::abs(value)>WIN/2||Clock::now()>=deadline)break;
        }
        return best;
    }
};
}

extern "C" int bs_choose(const int* data,int size,int ms,int depth,const double* weights,double* stats) {
    auto start=Clock::now();
    try {
        Config c;State s=decode(data,size,c);
        Search search(c,std::clamp(ms,1,400),weights);
        int result=search.run(s,std::clamp(depth,1,32));
        stats[0]=search.completed;stats[1]=double(search.nodes);stats[2]=search.value;
        stats[3]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return result;
    }catch(...){return -1;}
}

// Deterministic transition hook for differential tests. Food spawning belongs to
// the environment and is deliberately not predicted by the search.
extern "C" int bs_step(const int* data,int size,const int* moves,int* out,int capacity) {
    try {
        Config c;auto s=decode(data,size,c);std::vector<int> a(s.snakes.size());
        for(size_t i=0;i<a.size();i++){if(moves[i]<0||moves[i]>3)return -1;a[i]=moves[i];}
        auto t=step(s,a,c);int k=0;
        auto put=[&](int v){if(k>=capacity)throw std::runtime_error("capacity");out[k++]=v;};
        put(int(t.food.count()));for(int p=0;p<c.w*c.h;p++)if(t.food[p])put(p);
        put(int(t.snakes.size()));
        for(const auto& b:t.snakes){put(b.alive);put(std::max(0,b.health));put(int(b.body.size()));for(int p:b.body)put(p);}
        return k;
    }catch(...){return -1;}
}
