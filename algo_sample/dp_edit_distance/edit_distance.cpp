#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

int D(int i, int j, string s, string t)
{
    if(s[i-1] == t[j-1]) {
        return 0;
    }
    return 1;
}

int main()
{
    string s = "the";
    string t = "ten";
    int s_len = s.size();
    int t_len = t.size();

    int dp[4][4];
    for(int i = 0; i < 4; i++) {
        dp[i][0] = i;
    }
    for(int j = 0; j < 4; j++) {
        dp[0][j] = j;
    }

    for(int i = 1; i < 4; i++) {
        for(int j = 1; j < 4; j++) {
            dp[i][j] = min(
                dp[i-1][j]+1,
                min(dp[i][j-1]+1, dp[i-1][j-1]+D(i, j, s, t))
            );
        }
    }

    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            cout << " " << dp[i][j];
        }
        cout << endl;
    }

    return 0;
}