package src;

import java.util.*;

public class Lc1000_1099 {
    // 1032. Stream of Characters - H
    // 倒序建树, 倒序查找, 52ms
    class StreamChecker {
        class Trie {
            Trie[] ch = new Trie[26];
            boolean isEnd = false;
        }

        Trie root = new Trie();
        StringBuilder sb = new StringBuilder();

        void insert(String s) {
            Trie node = root;
            for (int i = s.length() - 1; i >= 0; --i) {
                int p = s.charAt(i) - 'a';
                if (node.ch[p] == null)
                    node.ch[p] = new Trie();
                node = node.ch[p];
            }
            node.isEnd = true;
        }

        public StreamChecker(String[] words) {
            for (String w : words)
                insert(w);
        }

        public boolean query(char letter) {
            sb.append(letter);
            int n = sb.length();
            Trie node = root;
            for (int i = n - 1; i >= Math.max(0, n - 200); --i) {
                int p = sb.charAt(i) - 'a';
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

        public boolean query2(char letter) {
            sb.append(letter);
            int n = sb.length();
            Trie node = root;
            for (int i = n - 1; i >= Math.max(0, n - 200) && node != null; --i) {
                int p = sb.charAt(i) - 'a';
                node = node.ch[p];
                if (node != null && node.isEnd) {
                    return true;
                }
            }
            return false;
        }
    }

    // 52ms
    class StreamChecker2 {
        class Trie {
            Trie[] ch;
            boolean isEnd;

            public Trie() {
                this.ch = new Trie[26]; // 注意 constructor 要 new 数组, 否则 default = null
            }
        }

        Trie root;
        StringBuilder sb = new StringBuilder();

        void insert(String word) {
            Trie node = root;
            for (int i = word.length() - 1; i >= 0; --i) {
                char c = word.charAt(i);
                if (node.ch[c - 'a'] == null) {
                    node.ch[c - 'a'] = new Trie();
                }
                node = node.ch[c - 'a'];
            }
            node.isEnd = true;
        }

        public StreamChecker2(String[] words) {
            root = new Trie();
            for (String w : words) {
                insert(w);
            }
        }

        public boolean query(char letter) {
            sb.append(letter);
            Trie node = root;
            for (int i = sb.length() - 1; i >= Math.max(0, sb.length() - 200); --i) {
                char c = sb.charAt(i);
                if (node.ch[c - 'a'] == null) {
                    return false;
                }
                node = node.ch[c - 'a'];
                if (node.isEnd) {
                    return true;
                }
            }
            return false;
        }
    }

    // 49ms
    class StreamChecker3 {
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
                for (int i = s.length() - 1; i >= 0; --i) {
                    int p = s.charAt(i) - 'a';
                    if (node.ch[p] == null) {
                        node.ch[p] = new TrieNode();
                    }
                    node = node.ch[p];

                    // As an optimization, this method truncates words, if one ends with the other.
                    // For example, if you add "ball" and "football", only "ball" is kept in the trie.
                    // But it does not seem too obvious
                    // if (node.isEnd) {
                    //     return;
                    // }

                }
                node.isEnd = true;
            }

            public boolean search(char[] ch, int index, int lmt) {
                TrieNode node = root;
                int j = index;
                for (int i = 0; i < lmt; ++i) {
                    if (--j < 0) { // 49ms
                        j += 200;
                    }

                    // j = --j < 0 ? j + 200 : j;     // 49ms
                    // j = ((--j % 200) + 200) % 200; // 54ms, modulo operation is slow

                    node = node.ch[ch[j] - 'a'];
                    if (node == null)
                        return false;
                    if (node.isEnd)
                        return true;
                }
                return false;
            }
        }

        Trie root;
        char[] queries = new char[200]; // rotating array, 200 char limit (for the stream of chars)
        int index = 0, lmt = 0;

        public StreamChecker3(String[] words) {
            root = new Trie();
            for (String w : words)
                root.insert(w);
        }

        public boolean query(char letter) {
            // Wrap around when we hit the end.
            queries[index] = letter;
            index = ++index % 200;
            lmt = lmt < 200 ? ++lmt : lmt;
            return root.search(queries, index, lmt);
        }
    }

}
