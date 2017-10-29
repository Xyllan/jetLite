package edu.nyu.jetlite;

public class WordInfo {
    public String word;
    public int distance1;
    public int distance2;
    public WordInfo(String word, int distance1, int distance2) {
        this.word = word;
        this.distance1 = distance1;
        this.distance2 = distance2;
    }
    @Override
    public String toString() {
        return word + " " + distance1 + " " + distance2;
    }
    public String encode() {
        return word + "}}}" + distance1 + "}}}" + distance2;
    }
}