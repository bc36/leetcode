package template_trick;

public class Trie2 {
    class Trie {
        Trie[] ch;
        boolean isEnd;

        public Trie() {
            this.ch = new Trie[26]; // 注意 constructor 要 new 数组, 否则 default = null
        }
    }

    Trie root;

    public void insert(String s) {
        Trie node = root;
        for (int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            if (node.ch[c - 'a'] == null) {
                node.ch[c - 'a'] = new Trie();
            }
            node = node.ch[c - 'a'];
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
