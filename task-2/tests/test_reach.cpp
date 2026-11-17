#include "../src/engine.cpp"
#include <cassert>
#include <deque>
#include <iostream>
#include <random>

// An independent cell queue is the oracle for the word-parallel bitboard flood.
int referenceSafeArea(const State& s,int who,const Config& c,
                      const std::array<int,CELLS>& release,const std::vector<Reach>& rr) {
    std::vector<int> distance(c.w*c.h,10000);
    std::deque<int> queue{s.snakes[who].body[0]};distance[queue[0]]=0;int area=0;
    while(!queue.empty()) {
        int p=queue.front();queue.pop_front();area++;
        for(int v:c.next[p]) {
            int nextDistance=distance[p]+1;
            if(v<0||distance[v]!=10000||release[v]>nextDistance)continue;
            bool allowed=true;
            for(size_t enemy=0;enemy<s.snakes.size();enemy++)if(int(enemy)!=who&&s.snakes[enemy].alive) {
                if(rr[enemy].dist[v]<nextDistance ||
                   (rr[enemy].dist[v]==nextDistance&&s.snakes[enemy].body.size()>=s.snakes[who].body.size()))
                    allowed=false;
            }
            if(allowed){distance[v]=nextDistance;queue.push_back(v);}
        }
    }
    return area;
}

int main() {
    std::mt19937 rng(72841);int cases=0;
    for(auto size:std::vector<std::pair<int,int>>{{2,2},{11,11},{8,16},{16,8},{25,5},{5,25},{13,9}})
    for(bool wrapped:{false,true})for(int sample=0;sample<1000;sample++) {
        Config c;c.w=size.first;c.h=size.second;c.wrapped=wrapped;c.init();State s;
        for(int i=0;i<2;i++) {
            Snake a;int length=1+rng()%40;
            for(int j=0;j<length;j++)a.body.push_back(int(rng()%(c.w*c.h)));
            s.snakes.push_back(a);
        }
        std::array<int,CELLS> release{};
        for(int p=0;p<c.w*c.h;p++) {
            release[p]=rng()%4==0?int(rng()%180):0;
            if(rng()%15==0)s.food.set(p);
        }
        BitFlood field(s,c,release);std::vector<Reach> rr;
        for(int i=0;i<2;i++) {
            auto expected=reach(s,i,c,release),actual=bitReach(s,i,c,field);
            assert(actual.area==expected.area);
            assert(actual.staticArea==expected.staticArea);
            assert(actual.food==expected.food);
            assert(actual.dist==expected.dist);
            rr.push_back(expected);
        }
        for(int i=0;i<2;i++)assert(safeBitArea(s,i,c,field,rr)==referenceSafeArea(s,i,c,release,rr));
        cases++;
    }
    std::cout<<cases<<" boards: reach and safe space match independent queue floods\n";
}
