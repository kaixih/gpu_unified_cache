#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <climits>

using namespace std;

static int warp_dim[3] = {5, 0, 0};
static int crs_fct = 4;
static int crs_dim = 0;
static int ghst_layers = 1;
static int warp_size = 32;
static int sten_dim = 1;

// static int warp_dim[3] = {5, 0, 0};
// static int crs_fct = 4;
// static int crs_dim = 1;
// static int ghst_layers = 1;
// static int warp_size = 32;
// static int sten_dim = 2;

// static int warp_dim[3] = {3, 2, 0};
// static int crs_fct = 4;
// static int crs_dim = 2;
// static int ghst_layers = 1;
// static int warp_size = 32;
// static int sten_dim = 3;

typedef pair<int,int> intPair;

int quick_pow(int a, int b)
{
    int ans = 1;
    while(b > 0)
    {
        if(b&1) ans = ans*a;
        a = a*a;
        b >>= 1;
    }
    return ans;
}

int getNumOfRegs()
{
    int ans = quick_pow(2, warp_dim[crs_dim])*crs_fct+2*ghst_layers;
    for(int i = crs_dim-1; i >= 0; i--)
    {
        ans *= quick_pow(2, warp_dim[i])+2*ghst_layers;
    }
    ans = (ans+warp_size-1)/warp_size;

    return ans;
}

void print_load_code(int num_regs)
{
    for(int i = 0; i < sten_dim; i++)
    {
        cout << "int new_id" << i << " ;" << endl;
    }

    for(int j = 0; j < num_regs - 1; j++)
    {
        for(int i = 0; i < sten_dim; i++)
        {
            cout << "new_id" << i << " = " ;
            cout << "(warp_id" << i << "<<" << warp_dim[i] << ") + " ;
            cout << "lane_id_it" ;
            int mul = 1;
            bool flag_mul = false;
            for(int k = i-1; k >= 0; k--)
            {
                mul *= quick_pow(2, warp_dim[k])+2*ghst_layers;
            }
            if(mul != 1)
            {
                cout << "/" << mul ;
                flag_mul = true;
            }
            if(!flag_mul || flag_mul && i != sten_dim-1)
                cout << "%" << quick_pow(2, warp_dim[i])+2*ghst_layers << " ;" << endl;
            else
                cout << " ;" << endl;

        }
        // cout << "reg" << j << " = " << "ACC_" << sten_dim << "D(in, " ;
        cout << "reg" << j << " = " << "IN_" << sten_dim << "D(" ;
        for(int i = 0; i < sten_dim; i++)
        {
            // cout << "new_id" << i ;
            cout << "new_id" << sten_dim-i-1 ;
            if(i != sten_dim-1)
                cout << ", " ;
            else
                cout << ") ;" << endl ;
        }
        
        cout << "lane_id_it += " << warp_size << " ;" << endl;
    }

    // deal with the tail case
    for(int i = 0; i < sten_dim; i++)
    {
        cout << "new_id" << i << " = " ;
        cout << "(warp_id" << i << "<<" << warp_dim[i] << ") + " ;
        cout << "lane_id_it" ;
        int mul = 1;
        bool flag_mul = false;
        for(int k = i-1; k >= 0; k--)
        {
            mul *= quick_pow(2, warp_dim[k])+2*ghst_layers;
        }
        if(mul != 1)
        {
            cout << "/" << mul ;
            flag_mul = true;
        }
        if(!flag_mul || flag_mul && i != sten_dim-1)
            cout << "%" << quick_pow(2, warp_dim[i])+2*ghst_layers << " ;" << endl;
        else
            cout << " ;" << endl;

    }
    for(int i = 0; i < sten_dim; i++)
    {
        cout << "new_id" << i << " = " ;
        cout << "(new_id" << i << " < dim" << i << "+" << 2*ghst_layers << ")? new_id";
        cout << i << " : dim" << i << "+" << 2*ghst_layers-1 << " ;" << endl ;
    }
    // cout << "reg" << num_regs-1 << " = " << "ACC_" << sten_dim << "D(in, " ;
    cout << "reg" << num_regs-1 << " = " << "IN_" << sten_dim << "D(" ;
    for(int i = 0; i < sten_dim; i++)
    {
        // cout << "new_id" << i ;
        cout << "new_id" << sten_dim-i-1 ;
        if(i != sten_dim-1)
            cout << ", " ;
        else
            cout << ") ;" << endl ;
    }


}

#define __VERBOSE

void calculate_friend_ids(vector<vector<vector<vector<intPair>>>> &friend_ids, int fct0, int fct1, int fct2)
{
    int id = 0;
    int ex_warp_dim0 = quick_pow(2, warp_dim[0])+2*ghst_layers;
    int ex_warp_dim1 = quick_pow(2, warp_dim[1])+2*ghst_layers;
    int ex_warp_dim2 = quick_pow(2, warp_dim[2])+2*ghst_layers;
    for(int c = 0; c < crs_fct; c++)
    {
#ifdef __VERBOSE
        cout << "// job" << c << ": ";
#endif
        for(int k = 0; k < fct2; k++)
        {
            if(crs_dim == 2 && k == 0) // coarsen along z-axis
            {
                int face = ex_warp_dim0*ex_warp_dim1;
                id = c * (ex_warp_dim2-2*ghst_layers) * face ;//% warp_size;
            }
            for(int j = 0; j < fct1; j++)
            {
                if(crs_dim == 1 && j == 0) // coarsen along y-axis 
                {
                    int line = ex_warp_dim0;
                    id = c * (ex_warp_dim1-2*ghst_layers) * line;// % warp_size;
                }
                for(int i = 0; i < fct0; i++)
                {

                    if(crs_dim == 0 && i == 0) // coarsen along x-axis 
                    {
                        int point = 1;
                        id = c * (ex_warp_dim0-2*ghst_layers) * point;// % warp_size;
                    }
                    friend_ids[c][k][j][i].first  = (id)%warp_size;
                    friend_ids[c][k][j][i].second = (id)/warp_size;
                    id++;
#ifdef __VERBOSE
                    cout << setw(2) << friend_ids[c][k][j][i].first << " " ;
#endif
                }
                id += (ex_warp_dim0 - fct0);
#ifdef __VERBOSE
                cout << " | " ;
#endif
            }
            id += (ex_warp_dim1-fct1)*ex_warp_dim0;
        }
        id += (ex_warp_dim2-fct2)*ex_warp_dim1*ex_warp_dim0;
#ifdef __VERBOSE
        cout << endl ;
#endif
    }

}

void print_friend_code(vector<vector<vector<vector<intPair>>>> &friend_ids, int job_id, int id0, int id1, int id2)
{
    cout << "friend_id" << job_id << " = (lane_id+" << setw(2) << friend_ids[job_id][id2][id1][id0].first;
    // assume warp dim only has two effective values in position 0 and 1
    if(quick_pow(2, warp_dim[1]) != 1)
        cout << "+((lane_id>>" << warp_dim[0] <<")*" << 2*ghst_layers << "))&" << warp_size-1 << " ;" ;
    else
        cout << ")&" << warp_size-1 << " ;" ;
    cout << endl;
}

void print_shfl_code(vector<vector<vector<vector<intPair>>>> &friend_ids, int job_id, int id0, int id1, int id2, string pre)
{
    cout << "tx" << job_id << " = __shfl(" << pre << "reg" << friend_ids[job_id][id2][id1][id0].second;
    cout << ", friend_id" << job_id << ");" << endl;
    int lb, rb;
    int ex_warp_dim0 = (sten_dim == 1)? INT_MAX: quick_pow(2, warp_dim[0]) + 2*ghst_layers;
    int previous_regs = friend_ids[job_id][id2][id1][id0].first + friend_ids[job_id][id2][id1][id0].second*warp_size;
    lb = previous_regs%ex_warp_dim0;
    rb = (lb + quick_pow(2, warp_dim[0]));
    int count0 = 1;
    int mark[2] = {-1, -1};
    char buf[2] = {'y', 'z'};
    int buf_id = 0;
    int reg_id = friend_ids[job_id][id2][id1][id0].second;
    bool reg_id_flag = false;
    // cout << "lb = " << lb << " ; rb = " << rb << endl;
    while(count0 < warp_size)
    {
        previous_regs++;
        int tmp_rw = (previous_regs%ex_warp_dim0);
        int tmp_wp = (previous_regs%warp_size);
        if(tmp_wp == 0)
        {
            reg_id++;
            reg_id_flag = true;
        }
        if(lb <= tmp_rw && tmp_rw < rb)
        {
            if(reg_id_flag)
            {
                cout << "t" << buf[buf_id] << job_id << " = __shfl(" << pre << "reg" << reg_id;
                cout << ", friend_id" << job_id << ");" << endl;
                mark[buf_id] = count0;
                buf_id++;
                reg_id_flag = false;
            }
            count0++;
        }
    }
    // cout << "return ";
    cout << "sum" << job_id << " += a" << "*(";
    if(mark[0] != -1 && mark[1] != -1)
    {
        cout << "(lane_id < " << mark[0] << " )? tx";
        cout << job_id << ": ((lane_id < " << mark[1] << ")? ty" << job_id << ": tz" << job_id << ")";
    } else if(mark[0] != -1 )
    {
        cout << "(lane_id < " << mark[0] << " )? tx";
        cout << job_id << ": ty" << job_id;
    } else 
    {
        cout << "tx" << job_id;
    }
    // cout << ";" << endl;
    cout << ");" << endl;

}


void print_excg_code(vector<vector<vector<vector<intPair>>>> &friend_ids, int fct0, int fct1, int fct2)
{
    for(int id2 = 0; id2 < fct2; id2++)
    {
        for(int id1 = 0; id1 < fct1; id1++)
        {
            for(int id0 = 0; id0 < fct0; id0++)
            {
                cout << "// process (" << id0 << ", " << id1 << ", " << id2 << ")" << endl;
                for(int job_id = 0; job_id < crs_fct; job_id++)
                {
                    print_friend_code(friend_ids, job_id, id0, id1, id2);
                    print_shfl_code(friend_ids, job_id, id0, id1, id2, "");
                    // print_shfl_code(friend_ids, job_id, id0, id1, id2, "t2_");
                    // print_shfl_code(friend_ids, job_id, id0, id1, id2, "t3_");
                    // getchar();
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    int num_regs = getNumOfRegs();
    cout << "// num_regs: " << num_regs << endl;
    for(int i = 0; i < num_regs; i++)
    {
        cout << "T reg" << i << " ;" << endl;
    }

    cout << "// load to regs: " << endl;
    print_load_code(num_regs);

    int fct0 = (crs_dim>=0)?1+2*ghst_layers:1;
    int fct1 = (crs_dim>=1)?1+2*ghst_layers:1;
    int fct2 = (crs_dim>=2)?1+2*ghst_layers:1;
    cout << "// neighbor list: " << fct2 << "*" << fct1 << "*" << fct0 << endl; 
    vector<vector<vector<vector<intPair>>>> friend_ids(crs_fct, 
            vector<vector<vector<intPair>>>(fct2, 
            vector<vector<intPair>>(fct1, 
            vector<intPair>(fct0))));
    calculate_friend_ids(friend_ids, fct0, fct1, fct2);

    print_excg_code(friend_ids, fct0, fct1, fct2);



}
