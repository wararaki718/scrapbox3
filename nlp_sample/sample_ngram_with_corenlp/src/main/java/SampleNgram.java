import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.StringUtils;

import java.util.ArrayList;
import java.util.List;

public class SampleNgram {
    public static void main(String[] args) {
        ArrayList<String> words = new ArrayList<String>(){
            {
                add("私");
                add("は");
                add("ご飯");
                add("を");
                add("食べ");
                add("ます");
                add("。");
            }
        };
        List<List<String>> ngrams = CollectionUtils.getNGrams(words, 3, 3);
        for(List<String> ngram: ngrams) {
            System.out.println(StringUtils.join(ngram, " "));
        }
        System.out.println("DONE!");
    }
}
