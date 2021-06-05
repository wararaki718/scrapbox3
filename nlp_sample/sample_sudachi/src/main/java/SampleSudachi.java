import com.worksap.nlp.sudachi.Dictionary;
import com.worksap.nlp.sudachi.DictionaryFactory;
import com.worksap.nlp.sudachi.Morpheme;
import com.worksap.nlp.sudachi.Tokenizer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class SampleSudachi {
    public static void main(String[] args) {
        Path path = Paths.get(SampleSudachi.class.getResource("sudachi.json").getPath());
        System.out.println(path.toString());

        List<String> tmp = null;
        try {
            tmp = Files.readAllLines(path);
        } catch(IOException e) {
            e.printStackTrace();
        }

        String settings = String.join("\n", tmp);
        Dictionary dict = null;
        try {
             dict = new DictionaryFactory().create(settings);
        } catch(IOException e) {
            e.printStackTrace();
        }

        Tokenizer tokenizer = dict.create();
        String text = "私はご飯を食べます。";
        List<Morpheme> tokens = tokenizer.tokenize(text);
        for(Morpheme token: tokens) {
            System.out.println(token.surface());
            System.out.println(token.normalizedForm());
            System.out.println(token.dictionaryForm());
            System.out.println(token.partOfSpeech());
            System.out.println();
        }

        System.out.println("DONE");
    }
}
