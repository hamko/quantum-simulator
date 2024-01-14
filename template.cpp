// CCX ゲートのテンソル表記での分解方法 https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.CCXGate

#include <bits/stdc++.h>
#include <sys/time.h>
using namespace std;

#define rep(i,n) for(long long i = 0; i < (long long)(n); i++)
#define repi(i,a,b) for(long long i = (long long)(a); i < (long long)(b); i++)
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define fi first
#define se second
#define mt make_tuple
#define mp make_pair
template<class T1, class T2> bool chmin(T1 &a, T2 b) { return b < a && (a = b, true); }
template<class T1, class T2> bool chmax(T1 &a, T2 b) { return a < b && (a = b, true); }

using ll = long long; using vll = vector<ll>; using vvll = vector<vll>; using P = pair<ll, ll>;
using comp = complex<double>;
using gate_t = vector<vector<comp>>;
ll ugauss(ll a, ll b) { if (!a) return 0; if (a>0^b>0) return a/b; else return (a+(a>0?-1:1))/b+1; }
ll lgauss(ll a, ll b) { if (!a) return 0; if (a>0^b>0) return (a+(a>0?-1:1))/b-1; else return a/b; }
template <typename T, typename U> ostream &operator<<(ostream &o, const pair<T, U> &v) {  o << "(" << v.first << ", " << v.second << ")"; return o; }
template<size_t...> struct seq{}; template<size_t N, size_t... Is> struct gen_seq : gen_seq<N-1, N-1, Is...>{}; template<size_t... Is> struct gen_seq<0, Is...> : seq<Is...>{};
template<class Ch, class Tr, class Tuple, size_t... Is>
void print_tuple(basic_ostream<Ch,Tr>& os, Tuple const& t, seq<Is...>){ using s = int[]; (void)s{0, (void(os << (Is == 0? "" : ", ") << get<Is>(t)), 0)...}; }
template<class Ch, class Tr, class... Args> 
auto operator<<(basic_ostream<Ch, Tr>& os, tuple<Args...> const& t) -> basic_ostream<Ch, Tr>& { os << "("; print_tuple(os, t, gen_seq<sizeof...(Args)>()); return os << ")"; }
ostream &operator<<(ostream &o, const vvll &v) { rep(i, v.size()) { rep(j, v[i].size()) o << v[i][j] << " "; o << endl; } return o; }
ostream &operator<<(ostream &o, const gate_t &v) { rep(i, v.size()) { o << "| "; rep(j, v[i].size()) o << v[i][j] << " "; o << "|" << endl; } return o; }
template <typename T> ostream &operator<<(ostream &o, const vector<T> &v) { o << '['; rep(i, v.size()) o << v[i] << (i != v.size()-1 ? ", " : ""); o << "]";  return o; }
template <typename T> ostream &operator<<(ostream &o, const deque<T> &v) { o << '['; rep(i, v.size()) o << v[i] << (i != v.size()-1 ? ", " : ""); o << "]";  return o; }
template <typename T>  ostream &operator<<(ostream &o, const set<T> &m) { o << '['; for (auto it = m.begin(); it != m.end(); it++) o << *it << (next(it) != m.end() ? ", " : ""); o << "]";  return o; }
template <typename T>  ostream &operator<<(ostream &o, const unordered_set<T> &m) { o << '['; for (auto it = m.begin(); it != m.end(); it++) o << *it << (next(it) != m.end() ? ", " : ""); o << "]";  return o; }
template <typename T, typename U>  ostream &operator<<(ostream &o, const map<T, U> &m) { o << '['; for (auto it = m.begin(); it != m.end(); it++) o << *it << (next(it) != m.end() ? ", " : ""); o << "]";  return o; }
template <typename T, typename U, typename V>  ostream &operator<<(ostream &o, const unordered_map<T, U, V> &m) { o << '['; for (auto it = m.begin(); it != m.end(); it++) o << *it; o << "]";  return o; }
vector<int> range(const int x, const int y) { vector<int> v(y - x + 1); iota(v.begin(), v.end(), x); return v; }
template <typename T> istream& operator>>(istream& i, vector<T>& o) { rep(j, o.size()) i >> o[j]; return i;}
template <typename T, typename S, typename U> ostream &operator<<(ostream &o, const priority_queue<T, S, U> &v) { auto tmp = v; while (tmp.size()) { auto x = tmp.top(); tmp.pop(); o << x << " ";} return o; }
template <typename T> ostream &operator<<(ostream &o, const queue<T> &v) { auto tmp = v; while (tmp.size()) { auto x = tmp.front(); tmp.pop(); o << x << " ";} return o; }
template <typename T> ostream &operator<<(ostream &o, const stack<T> &v) { auto tmp = v; while (tmp.size()) { auto x = tmp.top(); tmp.pop(); o << x << " ";} return o; }
template <typename T> unordered_map<T, ll> counter(vector<T> vec){unordered_map<T, ll> ret; for (auto&& x : vec) ret[x]++; return ret;};
void vizGraph(vvll& g, int mode = 0, string filename = "out.png") { ofstream ofs("./out.dot"); ofs << "digraph graph_name {" << endl; set<P> memo; rep(i, g.size())  rep(j, g[i].size()) { if (mode && (memo.count(P(i, g[i][j])) || memo.count(P(g[i][j], i)))) continue; memo.insert(P(i, g[i][j])); ofs << "    " << i << " -> " << g[i][j] << (mode ? " [arrowhead = none]" : "")<< endl;  } ofs << "}" << endl; ofs.close(); system(((string)"dot -T png out.dot >" + filename).c_str()); }
struct timeval start; double sec() { struct timeval tv; gettimeofday(&tv, NULL); return (tv.tv_sec - start.tv_sec) + (tv.tv_usec - start.tv_usec) * 1e-6; }
size_t random_seed; struct init_{init_(){ ios::sync_with_stdio(false); cin.tie(0); gettimeofday(&start, NULL); struct timeval myTime; struct tm *time_st; gettimeofday(&myTime, NULL); time_st = localtime(&myTime.tv_sec); srand(myTime.tv_usec); random_seed = RAND_MAX / 2 + rand() / 2; }} init__;
#define ldout fixed << setprecision(40) 

#define EPS (double)1e-6
#define INF (ll)1e18
#define mo  (ll)(1e9+7)

template <typename T> bool next_combination(const T first, const T last, int k) {
    const T subset = first + k;
    // empty container | k = 0 | k == n
    if (first == last || first == subset || last == subset) {
        return false;
    }
    T src = subset;
    while (first != src) {
        src--;
        if (*src < *(last - 1)) {
            T dest = subset;
            while (*src >= *dest) {
                dest++;
            }
            iter_swap(src, dest);
            rotate(src + 1, dest + 1, last);
            rotate(subset, subset + (last - dest) - 1, last);
            return true;
        }
    }
    // restore
    rotate(first, subset, last);
    return false;
}

enum op_mode {
    OP_X,
    OP_Y,
    OP_Z,
    OP_RY,
    OP_CX,
    NUM_OP
};
vll cost;

vector<comp> psi_cands;

using op_t = vll;
using circuit_t = vector<op_t>;
#define X (vector<ll>({OP_X, -1}))
#define Y (vector<ll>({OP_Y, -1}))
#define Z (vector<ll>({OP_Z, -1}))
#define RY (vector<ll>({OP_RY, -1, -1})) // target, pi/4*j
#define CX (vector<ll>({OP_CX, -1, -1})) // controlled, target

ll n; // input # of truth table (represented in 0 ~ n-1)
ll m; // output # of truth table (this can be arbitarary because any bit can be measured)
ll num_qubits; // # of all qubits

gate_t e; 
gate_t I_GATE; // in C^{2*2}
gate_t KETBRA00_GATE; // |0><0| in C^{2*2}
gate_t KETBRA11_GATE; // |1><1| in C^{2*2}
gate_t X_GATE; // in C^{2*2}
gate_t Y_GATE; // in C^{2*2}
gate_t Z_GATE; // in C^{2*2}
gate_t RY_GATE[8]; // RY[2 pi / 8 * i] iin C^{2*2}
vector<gate_t> X_memo; // in C^{2^num_qubits*2^num_qubits}
vector<gate_t> Y_memo; // in C^{2^num_qubits*2^num_qubits}
vector<gate_t> Z_memo; // in C^{2^num_qubits*2^num_qubits}
vector<vector<gate_t>> RY_memo; // [qubit index][2 pi / 8 * i] in C^{2^num_qubits*2^num_qubits}
vector<vector<gate_t>> CX_memo; // [controlled i][target j] in C^{2^num_qubits*2^num_qubits}
vector<vector<vector<gate_t>>> CCX_memo; // [controlled i][controlled j][target h] in C^{2^num_qubits*2^num_qubits}

ll calculate_circuit_cost(const circuit_t& c) {
    ll ret = 0;
    for (auto&& x : c) {
        ret += cost[x[0]];
    }
    return ret;
}

void draw(circuit_t c) {
    vector<string> bitmap(num_qubits, string(3*c.size()+6, '-'));
    rep(gate_id, c.size()) {
        // 3 + 3 * i 列に書き込む
        ll j = 3 + 3 * gate_id;
        ll type = c[gate_id][0];
        if (type == OP_X) {
            bitmap[c[gate_id][1]][j] = 'X';
        } else if (type == OP_Y) {
            bitmap[c[gate_id][1]][j] = 'Y';
        } else if (type == OP_Z) {
            bitmap[c[gate_id][1]][j] = 'Z';
        } else if (type == OP_RY) {
            bitmap[c[gate_id][1]][j] = '0' + c[gate_id][2];
        } else if (type == OP_CX) {
            ll cbit = c[gate_id][1];
            ll tbit = c[gate_id][2];
            bitmap[cbit][j] = 'o';
            bitmap[tbit][j] = '+';
            if (cbit > tbit) {
                repi(h, tbit+1, cbit) {
                    bitmap[h][j] = '|';
                }
            } else {
                repi(h, cbit+1, tbit) {
                    bitmap[h][j] = '|';
                }
            }
        } else {
            cerr << "not supported" << endl;
        }
    }
    rep(i, num_qubits) {
        cout << bitmap[i] << endl;
    }
}

void draw(vector<comp> c) {
    rep(bit, c.size()) {
        cout << bitset<3>(bit) << " : " << real(c[bit]) << " + " << imag(c[bit]) << "i" << endl;
    }
}

vector<comp> multiply(gate_t& a, vector<comp>& b) {
    assert(a.size() >= 1);
    assert(b.size() >= 1);
    assert(a[0].size() >= 1);

    vector<comp> ret(b.size());
    rep(i, a.size()) {
        assert(a[i].size() == b.size());
        rep(j, a[i].size()) {
            ret[i] += a[i][j] * b[j];
        }
    }
    return ret;
}

// tensor
gate_t multiply(const gate_t& A, const gate_t& B) {
    int aRows = A.size(), aCols = A[0].size();
    int bRows = B.size(), bCols = B[0].size();
    gate_t result(aRows * bRows, vector<comp>(aCols * bCols, comp(0, 0)));

    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < aCols; ++j) {
            for (int k = 0; k < bRows; ++k) {
                for (int l = 0; l < bCols; ++l) {
                    result[i * bRows + k][j * bCols + l] = A[i][j] * B[k][l];
                }
            }
        }
    }

    return result;
}


// 行列の積を計算する関数
gate_t multiply_matrix(const gate_t& A, const gate_t& B) {
    size_t aRows = A.size(), aCols = A[0].size();
    size_t bRows = B.size(), bCols = B[0].size();

    // 結果となる行列のサイズを確認
    assert(aCols == bRows);

    // 結果を格納する行列を初期化
    gate_t result(aRows, vector<complex<double>>(bCols, 0));

    // 行列の積を計算
    for (size_t i = 0; i < aRows; ++i) {
        for (size_t j = 0; j < bCols; ++j) {
            for (size_t k = 0; k < aCols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

// 行列の加算を計算する関数
gate_t add_matrix(const gate_t& A, const gate_t& B) {
    size_t aRows = A.size(), aCols = A[0].size();
    size_t bRows = B.size(), bCols = B[0].size();

    // 行列のサイズが同じであることを確認
    assert(aRows == bRows && aCols == bCols);

    // 結果を格納する行列を初期化
    gate_t result(aRows, vector<complex<double>>(aCols, 0));

    // 行列の加算を計算
    for (size_t i = 0; i < aRows; ++i) {
        for (size_t j = 0; j < aCols; ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }

    return result;
}

gate_t getU(circuit_t c) {
    gate_t tmp(1ll<<num_qubits, vector<comp>(1ll<<num_qubits));
    rep(i, 1ll<<num_qubits) tmp[i][i] = comp(1, 0);
//    cout << tmp << endl;
    rep(i, c.size()) {
        ll type = c[i][0];
        if (type == OP_X) {
            tmp = multiply_matrix(X_memo[c[i][1]], tmp);
        } else if (type == OP_Y) {
            tmp = multiply_matrix(Y_memo[c[i][1]], tmp);
        } else if (type == OP_Z) {
            tmp = multiply_matrix(Z_memo[c[i][1]], tmp);
        } else if (type == OP_RY) {
            tmp = multiply_matrix(RY_memo[c[i][1]][c[i][2]], tmp);
        } else if (type == OP_CX) {
            tmp = multiply_matrix(CX_memo[c[i][1]][c[i][2]], tmp);
        } else {
            cerr << "not supported" << endl;
        }
    }
    return tmp;
}

vector<vector<double>> target;

// i 番目の qubits で 1 が観測される確率
// TODO 一回の操作で全 i についての prob を取得できそう
double get_prob(vector<comp>& p, ll i, ll target = 1) { 
    assert(i >= 0);
    double prob = 0;
    rep(bit, p.size()) if (bit & (1ll << i)) {
        double tmp = abs(p[bit]);
        prob += tmp * tmp;
    }
    return target ? prob : 1.0 - prob;
}

void init(void) {
    cout << "############################# INIT ##########################" << endl;
    e.resize(1); e[0].resize(1); e[0][0] = 1;
    cost.resize(NUM_OP);

    I_GATE.resize(2); I_GATE[0].resize(2); I_GATE[1].resize(2);
    I_GATE[0][0] = comp(1, 0);
    I_GATE[1][1] = comp(1, 0);

    KETBRA00_GATE.resize(2); KETBRA00_GATE[0].resize(2); KETBRA00_GATE[1].resize(2);
    KETBRA00_GATE[0][0] = comp(1, 0);

    KETBRA11_GATE.resize(2); KETBRA11_GATE[0].resize(2); KETBRA11_GATE[1].resize(2);
    KETBRA11_GATE[1][1] = comp(1, 0);

    X_GATE.resize(2); X_GATE[0].resize(2); X_GATE[1].resize(2);
    X_GATE[0][1] = comp(1, 0);
    X_GATE[1][0] = comp(1, 0);
    cost[OP_X] = 1;

    Y_GATE.resize(2); Y_GATE[0].resize(2); Y_GATE[1].resize(2);
    Y_GATE[0][1] = -comp(0, 1);
    Y_GATE[1][0] = comp(0, 1);
    cost[OP_Y] = 1;

    Z_GATE.resize(2); Z_GATE[0].resize(2); Z_GATE[1].resize(2);
    Z_GATE[0][0] = comp(1, 0);
    Z_GATE[1][1] = comp(-1, 0);
    cost[OP_Z] = 1;

    rep(i, 8) {
        double theta = (2. * M_PI) * i / 8;
        RY_GATE[i].resize(2); RY_GATE[i][0].resize(2); RY_GATE[i][1].resize(2);
        RY_GATE[i][0][0] = comp(cos(theta), 0);     RY_GATE[i][0][1] = comp(-sin(theta), 0);
        RY_GATE[i][1][0] = comp(sin(theta), 0);     RY_GATE[i][1][1] = comp(cos(theta), 0);
        cout << RY_GATE[i] << endl;
        cost[OP_RY] = 1;
    }

    X_memo.resize(num_qubits);
    rep(i, num_qubits) {
        gate_t tmp = e;
        rep(j, num_qubits) {
            tmp = multiply((i == j ? X_GATE : I_GATE), tmp);
        }
        X_memo[i] = tmp;
    }

    Y_memo.resize(num_qubits);
    rep(i, num_qubits) {
        gate_t tmp = e;
        rep(j, num_qubits) {
            tmp = multiply((i == j ? Y_GATE : I_GATE), tmp);
        }
        Y_memo[i] = tmp;
    }

    Z_memo.resize(num_qubits);
    rep(i, num_qubits) {
        gate_t tmp = e;
        rep(j, num_qubits) {
            tmp = multiply((i == j ? Z_GATE : I_GATE), tmp);
        }
        Z_memo[i] = tmp;
    }

    RY_memo.resize(num_qubits);
    rep(i, num_qubits) {
        RY_memo[i].resize(8);
    }
    rep(i, num_qubits) {
        rep(rot, 8) {
            gate_t tmp = e;
            rep(j, num_qubits) {
                tmp = multiply((i == j ? RY_GATE[rot] : I_GATE), tmp);
            }
            cout << i << " " << rot << " " << tmp << endl;
            RY_memo[i][rot] = tmp;
        }
    }

    CX_memo.resize(num_qubits);
    rep(i, num_qubits) {
        CX_memo[i].resize(num_qubits);
    }
    rep(i, num_qubits) rep(j, num_qubits) if (i != j) { // i: control bit, j: target bit
        // CNOT(0, 2) = 
        // I otimes I otimes |0><0| +
        // X otimes I otimes |1><1|
        gate_t tmp1 = e;
        rep(h, num_qubits) {
            tmp1 = multiply((h == i ? KETBRA00_GATE : I_GATE), tmp1);
        }

        gate_t tmp2 = e;
        rep(h, num_qubits) {
            if (h == i) {
                tmp2 = multiply(KETBRA11_GATE, tmp2);
            } else if (h == j) {
                tmp2 = multiply(X_GATE, tmp2);
            } else {
                tmp2 = multiply(I_GATE, tmp2);
            }
        }

        CX_memo[i][j] = add_matrix(tmp1, tmp2);
    }
    cost[OP_CX] = 10;

    // psi が取りうる複素数
    {
        psi_cands.push_back(comp(0,0));

        psi_cands.push_back(comp(+sqrt(2)/2,0));
        psi_cands.push_back(comp(-sqrt(2)/2,0));
        psi_cands.push_back(comp(0,+sqrt(2)/2));
        psi_cands.push_back(comp(0,-sqrt(2)/2));

        psi_cands.push_back(comp(+sqrt(2)/4,0));
        psi_cands.push_back(comp(-sqrt(2)/4,0));
        psi_cands.push_back(comp(0,+sqrt(2)/4));
        psi_cands.push_back(comp(0,-sqrt(2)/4));

        psi_cands.push_back(comp(+0.5,0));
        psi_cands.push_back(comp(-0.5,0));
        psi_cands.push_back(comp(0,+0.5));
        psi_cands.push_back(comp(0,-0.5));

        psi_cands.push_back(comp(+0.25,0));
        psi_cands.push_back(comp(-0.25,0));
        psi_cands.push_back(comp(0,+0.25));
        psi_cands.push_back(comp(0,-0.25));

        psi_cands.push_back(comp(+1,0));
        psi_cands.push_back(comp(0,+1));
        psi_cands.push_back(comp(-1,0));
        psi_cands.push_back(comp(0,-1));
    }
}

// 完全にマッチする qubits が見つかった時のみ
//     matching_target_id[output i] = i 番目の論理式に qubit j が一致
// を出力する
map<ll, ll> get_matching_ids(circuit_t c, gate_t circuitU = {}) {
    if (!circuitU.size()) {
        circuitU = getU(c);
    }

    // result[i][bit] = bit を入力したときに qubit i で 1 が観測される確率
    vector<vector<double>> result(num_qubits, vector<double>(1ll << n, 0)); 
    rep(bit, 1ll << n) {
        vector<comp> psi; // 2^qubits
        psi.resize(1ll << num_qubits, comp(0, 0));
        psi[bit] = 1;

        auto psi_ret = multiply(circuitU, psi);
        ll ret = -1;
        vector<double> truth_table(num_qubits);
        rep(i, num_qubits) {
            truth_table[i] = get_prob(psi_ret, i);
        }
        // 0 bit 目の答えが左として出力されるため注意。ほかの bitset などの表記では 0 bit 目が右として出力される
        rep(i, num_qubits) {
            result[i][bit] = truth_table[i];
            if (abs(abs(result[i][bit]) - 1.0) < EPS) {
                result[i][bit] = 1;
            }
            if (abs(result[i][bit]) < EPS) {
                result[i][bit] = 0;
            }
        }
    }
    /*
    draw(c);
    rep(i, num_qubits) {
        cout << "qubit " << i << " : " << result[i] << endl;
    }
    rep(j, m) {
        cout << "target : " << target[j] << endl;
    }
    */
    map<ll, ll> matching_target_id; 
    rep(i, num_qubits) {
        rep(j, m) {
            if (result[i] == target[j]) {
                matching_target_id[j] = i;
                break;
            }
        }
    }
    ll tef = 0;
    if (matching_target_id.size() != m) {
        return map<ll, ll>();
    } else {
        return matching_target_id;
    }
}

void test(void) {
    cout << "############################# TEST ##########################" << endl;
    vector<comp> p(4);
    p[0] = comp(0, 0);       // ket{00} = 0
    p[1] = comp(0.7071, 0);  // ket{01} = 1 / sqrt(2)
    p[2] = comp(0.7071, 0);  // ket{10} = 1 / sqrt(2)
    p[3] = comp(0, 0);       // ket{11} = 0

    // p の右から 0 ビット目に p をかける
    // (0, -1/sqrt(2), 1/sqrt(2), 0) になるはず
    gate_t tmpI = I_GATE;
    gate_t tmpZ = Z_GATE;
    gate_t ret = multiply(tmpI, tmpZ);
    cout << 
        multiply(
                ret, p 
                )
        << endl;


    // (1) otimes I_2 = I_2 になるはず
    cout << "1 otimes I_2" << endl;
    gate_t tmp; tmp.resize(1); tmp[0].resize(1); tmp[0][0] = 1;
    cout << multiply(tmp, I_GATE) << endl;

    // ket{00...00} で初期化しているので、確率は全て 1 になるはず
    vector<comp> psi; // 2^qubits
    psi.resize(1ll << num_qubits, comp(0, 0));
    psi[0] = 1; // 全て ket{0} で初期化
    cout << num_qubits << " " << psi << "#psi" << endl;
    rep(i, num_qubits) {
        cout << get_prob(psi, i) << endl; // デフォルト 0 がでる確率
    }

    {
        circuit_t c;
        op_t op1 = X; op1[1] = 0;
        c.push_back(op1);
        draw(c);
        gate_t circuitU = getU(c);
        cout << "U" << endl << circuitU << endl;
        cout << multiply(circuitU, psi) << endl;
        // ket{100} のみが 1+0i
        // つまり \ket{000} を入力したら qubit 0 = 1, qubit 1 = 0, qubit 2 = 0 が観測される。
        draw(multiply(circuitU, psi)); 
    }

    // X ゲート
    {
        circuit_t c;
        c.push_back(vector<ll>({OP_X, 2}));
        draw(c);
        gate_t circuitU = getU(c);
        cout << "U" << endl << circuitU << endl;

        rep(bit, 1 << 3) {
            cout << "# X TEST " << bitset<3>(bit) << endl;
            vector<comp> psi; // 2^qubits
            psi.resize(1ll << num_qubits, comp(0, 0));
            psi[bit] = 1;
            cout << "input" << endl;
            draw(psi);
            cout << "output" << endl;
            draw(multiply(circuitU, psi));
        }
    }


    // RY ゲート
    {
        circuit_t c;
        c.push_back(vector<ll>({OP_RY, 2, 1}));
        draw(c);
        gate_t circuitU = getU(c);
        cout << "U" << endl << circuitU << endl;

        rep(bit, 1 << 3) {
            cout << "# RY TEST " << bitset<3>(bit) << endl;
            vector<comp> psi; // 2^qubits
            psi.resize(1ll << num_qubits, comp(0, 0));
            psi[bit] = 1;
            cout << "input" << endl;
            draw(psi);
            cout << "output" << endl;
            draw(multiply(circuitU, psi));
        }
    }

    // RY ゲート (rot 1 x 2)
    {
        circuit_t c;
        c.push_back(vector<ll>({OP_RY, 2, 1}));
        c.push_back(vector<ll>({OP_RY, 2, 1}));
        draw(c);
        gate_t circuitU = getU(c);
        cout << "U" << endl << circuitU << endl;

        vector<comp> psi; // 2^qubits
        psi.resize(1ll << num_qubits, comp(0, 0));
        psi[1] = 1;

        cout << "input" << endl;
        draw(psi);
        cout << "output" << endl;
        draw(multiply(circuitU, psi));
    }

    // RY ゲート (rot 1 x 2)
    {
        circuit_t c;
        c.push_back(vector<ll>({OP_RY, 2, 2}));
        draw(c);
        gate_t circuitU = getU(c);
        cout << "U" << endl << circuitU << endl;

        vector<comp> psi; // 2^qubits
        psi.resize(1ll << num_qubits, comp(0, 0));
        psi[1] = 1;

        cout << "input" << endl;
        draw(psi);
        cout << "output" << endl;
        draw(multiply(circuitU, psi));
    }

    // なんちゃって CCX ゲート が Toffoli と一致しない。。
    // qiskit でも一致しないので、https://www.youtube.com/watch?v=UcgQXpFNubs が間違っているのだと思う
    {
        circuit_t c;
        c.push_back(vector<ll>({OP_RY, 2, 1}));
        c.push_back(vector<ll>({OP_CX, 1, 2}));
        c.push_back(vector<ll>({OP_RY, 2, 1}));
        c.push_back(vector<ll>({OP_CX, 0, 2}));
        c.push_back(vector<ll>({OP_RY, 2, 7}));
        c.push_back(vector<ll>({OP_CX, 1, 2}));
        c.push_back(vector<ll>({OP_RY, 2, 7}));
        draw(c);
        gate_t circuitU = getU(c);
        cout << "U" << endl << circuitU << endl;

        rep(bit, 1 << 3) {
            cout << "# pseudo CCX TEST " << bitset<3>(bit) << endl;
            vector<comp> psi; // 2^qubits
            psi.resize(1ll << num_qubits, comp(0, 0));
            psi[bit] = 1;
            cout << "input" << endl;
            draw(psi);
            cout << "output" << endl;
            draw(multiply(circuitU, psi));
        }
//        exit(0);
    }
    {
        circuit_t c;
        c.push_back(vector<ll>({OP_X, 0, 1}));
        c.push_back(vector<ll>({OP_RY, 1, 1}));
        c.push_back(vector<ll>({OP_CX, 0, 1}));
        c.push_back(vector<ll>({OP_RY, 1, 3}));
        draw(c);
        gate_t circuitU = getU(c);
        cout << "U" << endl << circuitU << endl;

        rep(bit, 1 << 2) {
            cout << "# pseudo OR TEST " << bitset<3>(bit) << endl;
            vector<comp> psi; // 2^qubits
            psi.resize(1ll << num_qubits, comp(0, 0));
            psi[bit] = 1;
            cout << "input" << endl;
            draw(psi);
            cout << "output" << endl;
            draw(multiply(circuitU, psi));
        }
        cout << get_matching_ids(c) << endl;
//        exit(0);
    }

}

void search_truth_table(void) {
    circuit_t c_init;
    queue<circuit_t> q;
    q.push(c_init);
    circuit_t ret_c;
    map<ll, ll> ret_ids;
    while (q.size()) {
        cout << "################ TRY" << endl;
        auto c = q.front();
        draw(c);
        q.pop();
        auto ids = get_matching_ids(c);
        cout << ids << " IDS" << endl;
        if (ids.size()) { // found
            ret_c = c;
            ret_ids = ids;
            cout << "#### circuit ####" << endl;
            cout << "cost : " << calculate_circuit_cost(ret_c) << endl;
            draw(ret_c);
            for (auto x : ret_ids) {
                cout << "qubit " << x.se << " corresponds to output " << x.fi << endl;
            }
            //            break;
        }
        // transitions
        for (auto gate : {X, Y, Z}) {
            rep(i, num_qubits) {
                op_t op = gate; op[1] = i;
                c.push_back(op);
                q.push(c);
                c.pop_back();
            }
        }
        rep(i, num_qubits) repi(j, i+1, num_qubits) {
            op_t op = CX; op[1] = i; op[2] = j;
            c.push_back(op);
            q.push(c);
            c.pop_back();
        }
    }
}

using encoded_psi_t = vector<ll>;
encoded_psi_t encode(vector<comp> psi) {
    vector<P> pos_and_state;
    rep(i, psi.size()) if (abs(psi[i]) > EPS) {
        comp c(1, 0);
        rep(psi_id, psi_cands.size()) {
            double dist = abs(psi[i] - psi_cands[psi_id]);
            if (dist < EPS) {
                pos_and_state.push_back(P(i, psi_id));
                break;
            }
        }
    }
    sort(all(pos_and_state));
    vll ret;
    for (auto&& x : pos_and_state) {
        ret.push_back(x.fi);
        ret.push_back(x.se);
    }
    if (ret.size() == 0) {
        cout << "failed to encode " << psi << endl;
        exit(1);
    }
    return ret;
}

// 一応全加算器 (4 qubits, 3 input, 2 output でも 14520000 で終了する
// NOT FOUND で終わるけど…（この遷移だけでは存在しないということなのだと思う）
// 2 1 4
// 0 0 | 0
// 0 1 | 0
// 1 0 | 0
// 1 1 | 1
priority_queue<pair<ll, circuit_t>, vector<pair<ll, circuit_t>>, greater<pair<ll, circuit_t>>> q;
map<vector<encoded_psi_t>, pair<ll, circuit_t>> memo;
void dijkstra(void) {
    circuit_t c_init;
    q.push(mp(0, c_init));
    // TODO ここに memo push 忘れてない？
    
    ll counter = 0;
    bool found = 0;
    while (q.size()) {
        counter++;

        auto tmp_now = q.top(); q.pop();
        auto cost_now = tmp_now.fi;
        auto c_now = tmp_now.se;
        auto U = getU(c_now);

        vector<encoded_psi_t> state;
        rep(bit, 1ll << n) {
            vector<comp> psi(1 << num_qubits);
            psi[bit] = 1;
            psi = multiply(U, psi);
            state.push_back(encode(psi));
            if (counter % 5000 == 0) {
            cout << "# TRIAL" << " " << counter << endl;
                cout << bitset<4>(bit) << " " << psi << " " << encode(psi) << endl;
            }
        }

        if (counter % 5000 == 0) {
            cout << "cost " << cost_now << endl;
            cout << "state " << endl << state << endl;
            draw(c_now);
        }


        if (memo.count(state)) {
            continue;
        }
        memo[state] = mp(cost_now, c_now);
        
        // Check if the state is the end
        auto ids = get_matching_ids(c_now, U);
        if (ids.size()) { // found
            cout << "FOUND" << endl;
            cout << "cost: " << cost << endl;
            draw(c_now);
            for (auto x : ids) {
                cout << "qubit " << x.se << " corresponds to output " << x.fi << endl;
            }
            found = 1;
            break; // end the dijkstra search
        }

        // transitions
        // X
        {
            rep(i, num_qubits) {
                if (c_now.size() && c_now.back()[0] == OP_X && c_now.back()[1] == i) continue; // 2 連続で X する意味はない

                op_t op = X; op[1] = i;
                c_now.push_back(op);
                q.push(mp(cost_now + cost[OP_X], c_now));
                c_now.pop_back();
            }
        }
        // RY
        {
            ll prev_ry_pos = -1;
            rep(i, c_now.size()) {
                if (c_now[i][0] == OP_RY) {
                    prev_ry_pos = c_now[i][1];
                    break;
                }
            }
            rep(i, num_qubits) {
                if (prev_ry_pos != -1 && prev_ry_pos != i) continue; // 探索の方針として、RY は単一の qubit にしかおけないということにしている。
                if (c_now.size() && c_now.back()[0] == OP_RY && c_now.back()[1] == i) continue; // 2 連続で RY する意味はない

                repi(rot, 1, 8) {
                    op_t op = RY; op[1] = i; op[2] = rot;
                    c_now.push_back(op);
                    q.push(mp(cost_now + cost[OP_RY], c_now));
                    c_now.pop_back();
                }
            }
        }
        // CX
        {
            rep(i, num_qubits) repi(j, i+1, num_qubits) {
                op_t op = CX; op[1] = i; op[2] = j;
                c_now.push_back(op);
                q.push(mp(cost_now + cost[OP_CX], c_now));
                c_now.pop_back();
            }
        }
    }
    if (!found) {
        cout << "NOT FOUND" << endl;
    }
    exit(0);
}

int main(void) {
    cin >> n >> m >> num_qubits;
    assert(n >= 1);
    assert(m >= 1);
    assert(m <= num_qubits);
    target.resize(m);

    rep(bit, 1ll << n) {
        string tmp; 
        rep(i, n) {
            cin >> tmp;
        }
        cin >> tmp;
        rep(j, m) {
            ll b; cin >> b;
            target[j].push_back(b);

        }
    }

    init();
    test();

    cout << "############################# MAIN ##########################" << endl;
    //    search_truth_table();
    dijkstra();

    return 0;
}
