#include <iostream>
#include <unordered_map>
#include <string>

int main()
{
    std::unordered_map<std::string, std::string> dic = {{"今日", "名詞"}, {"は", "助詞"}, {"天気", "名詞"}, {"よい", "形容詞"}, {"です", "助動詞"}};
    const std::string input = "今日はよい天気です.";
    for(size_t i = 0; i < input.size(); i++) {
        for(size_t n = i+1; n < input.size(); n++) {
            const std::string query = input.substr(i, n-i);
            if(dic.find(query) != dic.end()) {
                std::cout << query << " が見つかりました。" << std::endl;
            }
        }
    }
    
    return 0;
}