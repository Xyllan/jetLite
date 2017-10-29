package edu.nyu.jetlite;

import java.util.List;
import java.util.stream.Collectors;

public class RelationExample {
    public List<WordInfo> features;
    public String relation;
    public RelationExample(List<WordInfo> features, String relation) {
        this.features = features;
        this.relation = relation;
    }
    @Override
    public String toString() {
        return relation + "|||" + features.stream().map(WordInfo::encode).collect(Collectors.joining("|||"));
    }
}