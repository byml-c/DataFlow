#include <bits/stdc++.h>
#include <cstdio>

#define group_type vector<pair<double, double>>
using namespace std;

struct GROUP{
    group_type points;
    pair<double, double> center;
};

// 打印结果
void Print(vector<GROUP> &groups){
    for(int i=0, l=groups.size();i<l;++i){
        cout<<"Group "<<i+1<<":\n";
        for(int j=0, l2=groups[i].points.size();j<l2;++j){
            cout<<groups[i].points[j].first<<' '<<groups[i].points[j].second<<'\n';
        }
    }
}
// 计算两点间距离，使用欧氏距离的平方
inline double distance(pair<double, double> a, pair<double, double> b){
    return (a.first-b.first)*(a.first-b.first) + (a.second-b.second)*(a.second-b.second);
}
vector<GROUP> Kmeans(group_type &points, int k){
    vector<int> center;
    int size_p = points.size();

    // 如果点数小于等于k，直接返回
    if(size_p <= k){
        vector<GROUP> res;
        for(int i=0;i<size_p;++i){
            res.push_back((GROUP){group_type({points[i]}), points[i]});
        }
        return res;
    }
    
    // 随机选取k个点作为初始中心
    for(int i=0;i<k;++i){
        center.push_back(rand()%size_p);
        for(int j=0;j<i;++j){
            if(center[i] == center[j]){
                center.pop_back();
                i--;
                break;
            }
        }
    }

    vector<int> new_center;
    vector<vector<int>> groups;
    while(true){
        groups.clear();
        new_center.clear();

        // 将所有点划分到组
        for(int i=0;i<k;++i){
            groups.push_back(vector<int>{center[i]});
        }
        for(int i=0;i<size_p;++i){
            // 跳过中心点
            bool flag = false;
            for(int j=0;j<k;++j){
                if(i == center[j]){
                    flag = true;
                }
            }
            if(flag) continue;

            int min_group = 0;
            double min_dis = distance(points[i], points[center[0]]);
            for(int j=1;j<k;++j){
                double dis = distance(points[i], points[center[j]]);
                if(dis < min_dis){
                    min_dis = dis;
                    min_group = j;
                }
            }
            groups[min_group].push_back(i);
        }

        // 找到每组的中心（到其他点距离平方和最小的点）
        for(int i=0;i<k;++i){
            int size_ng = groups[i].size();
            double min_sum = 0;
            int min_index = 0;
            for(int j=0;j<size_ng;++j){
                double sum = 0;
                for(int l=0;l<size_ng;++l){
                    sum += distance(points[groups[i][j]], points[groups[i][l]]);
                }
                if(j == 0 || sum < min_sum){
                    min_sum = sum;
                    min_index = groups[i][j];
                }
            }
            new_center.push_back(min_index);
        }

        // 如果中心不再变化，结束
        bool same = true;
        for(int i=0;i<k;++i){
            if(new_center[i] != center[i]){
                same = false;
                break;
            }
        }
        if(same) break;
        else center = new_center;
    }
    
    vector<GROUP> result;
    for(int i=0;i<k;++i){
        group_type group;
        for(int j=0, l=groups[i].size();j<l;++j){
            group.push_back(points[groups[i][j]]);
        }
        result.push_back((GROUP){group, points[center[i]]});
    }
    return result;
}
double CalSSE(group_type &group, pair<double, double> center){
    double SSE = 0;
    int size_g = group.size();
    for(int i=0;i<size_g;++i){
        SSE += distance(group[i], center);
    }
    return SSE;
}
vector<GROUP> BiKmeans(group_type &points, int k){
    vector<GROUP> groups;
    int size_p = points.size();

    // 寻找初始中心点
    int center_idx = 0;
    double center_SSE = 0;
    for(int i=0;i<size_p;++i){
        double SSE = CalSSE(points, points[i]);
        if(i == 0 || SSE < center_SSE){
            center_idx = i;
            center_SSE = SSE;
        }
    }
    groups.push_back((GROUP){points, points[center_idx]});
    
    while(true){
        int size_g = groups.size();
        if(size_g >= k) break;

        int max_idx = 0;
        double max_SSE = 0;
        for(int i=0;i<size_g;++i){
            double SSE = CalSSE(groups[i].points, groups[i].center);
            if(i == 0 || SSE > max_SSE){
                max_idx = i;
                max_SSE = SSE;
            }
        }

        if(groups[max_idx].points.size() <= 1) break;
        
        auto res = Kmeans(groups[max_idx].points, 2);
        double SSE1 = CalSSE(res[0].points, res[0].center);
        double SSE2 = CalSSE(res[1].points, res[1].center);
        if(SSE1 + SSE2 < max_SSE){
            groups[max_idx] = (GROUP){res[0].points, res[0].center};
            groups.push_back((GROUP){res[1].points, res[1].center});
        }else break;
    }
    return groups;
}
int main(){
    int n, k;
    group_type points;

    cin>>n>>k;
    for(int i=1;i<=n;++i){
        double x, y;
        cin>>x>>y;
        points.push_back(make_pair(x, y));
    }
    auto res = BiKmeans(points, k);
    Print(res);
}