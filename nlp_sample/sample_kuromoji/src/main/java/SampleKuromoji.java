import com.atilika.kuromoji.unidic.Token;
import com.atilika.kuromoji.unidic.Tokenizer;

import java.util.List;

public class SampleKuromoji {
    public static void main(String[] args) {
        String s = "私はご飯を食べる。";
        Tokenizer tokenizer = new Tokenizer();

        List<Token> tokens = tokenizer.tokenize(s);
        for(Token token: tokens) {
            System.out.println(token.getSurface());
            System.out.println(token.getLemma());
            System.out.println(token.getPartOfSpeechLevel1());
            System.out.println();
        }

        System.out.println("DONE");
    }
}
