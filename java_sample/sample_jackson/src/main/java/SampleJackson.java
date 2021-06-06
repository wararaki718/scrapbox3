import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;


public class SampleJackson {
    public static void main(String[] args) {
        Path path = Paths.get(SampleJackson.class.getResource("sample.json").getPath());
        System.out.println(path);

        ObjectMapper objectMapper = new ObjectMapper();
        ArrayList<User> users = new ArrayList<>();
        try {
            BufferedReader in = new BufferedReader(new FileReader(path.toString()));
            while(in.ready()) {
                String json = in.readLine();
                User user = objectMapper.readValue(json, User.class);
                users.add(user);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        for(User user: users) {
            System.out.println("id="+user.id + ": name=" + user.name);
        }
        System.out.println("DONE");
    }
}
