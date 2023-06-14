package template_trick;

public class Trie3 {
    class TrieNode {
        TrieNode[] ch;
        boolean isEnd = false;

        public TrieNode() {
            ch = new TrieNode[26];
        }
    }

    class Trie {
        TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        public void insert(CharSequence s) {
            TrieNode node = root;
            for (int i = 0; i < s.length(); ++i) {
                int p = s.charAt(i) - 'a';
                if (node.ch[p] == null) {
                    node.ch[p] = new TrieNode();
                }
                node = node.ch[p];

            }
            node.isEnd = true;
        }

        public boolean query(char letter) {
            TrieNode node = root;
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

    Trie root;

    class Solution {
        public void Solve(String[] words) {
            root = new Trie();
            for (String w : words)
                root.insert(w);

            // do sth

            return;
        }
    }
}
