package template_trick;

public class Trie1 {
    class Trie {
        Trie[] ch = new Trie[26];
        boolean isEnd = false;
    }

    Trie root = new Trie();

    public void insert(String s) {
        Trie node = root;
        for (int i = 0; i < s.length(); ++i) {
            int p = s.charAt(i) - 'a';
            if (node.ch[p] == null)
                node.ch[p] = new Trie();
            node = node.ch[p];
        }
        node.isEnd = true;
    }

    public boolean query(char letter) {
        Trie node = root;
        for (int i = 0; i < 1; ++i) {
            
            // do sth

            int p = i;
            if (node.ch[p] == null) {
                return false;
            }
            node = node.ch[p];
            if (node.isEnd) {
                return true;
            }
        }
        return false;
    }
}
