// -*- tab-width: 4 -*-
// Title:         JetLite
// Version:       1.00
// Copyright (c): 2017
// Author:        Ralph Grishman
// Description:   A lightweight Java-based Information Extraction Tool

package edu.nyu.jetlite;

import edu.nyu.jetlite.tipster.*;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import edu.nyu.jet.aceJet.*;

/**
 *  A relation tagger trained on the ACE 2005 data.
 */

public class RelationTagger extends Annotator implements AutoCloseable {

    static boolean printProgress = true;

    public static String OTHER = "OTHER";

    // the detail with which to classify a relation
    public enum TypeDetail { Basic, Subtype, SubtypeWithOrder }

    // the file containing the MaxEnt model
    String modelFileName;

    // the MaxEnt model
    MaxEntModel model;

    // boolean flag to set if Tensorflow model is to be used
    boolean useTFmodel = true;

    // Java wrapper for the deep learning model
    TFRelationTagger tfTagger;

    // The detail with which to classify a relation
    TypeDetail typeDetail;

    /**
     *  Create a new RelationTagger.
     *
     *  @param  config  A jet property file.  Property RelationTagger.model.fileName
     *                 specifies the file to contain the model.
     */
    public RelationTagger (Properties config) throws IOException {
    this(config, false, TypeDetail.Basic);
    }


    public RelationTagger (Properties config, boolean useTFmodel, TypeDetail typeDetail) throws IOException {
    this.useTFmodel = useTFmodel;
    this.typeDetail = typeDetail;
	modelFileName = config.getProperty("RelationTagger.model.fileName");
	if(useTFmodel) {
        tfTagger = new TFRelationTagger(typeDetail);
    } else {
        model = new MalletMaxEntModel(modelFileName, "RelationTagger");
    }
    }

    public void close() {
        if(tfTagger != null) {
            tfTagger.close();
        }
        tfTagger = null;
    }

    /**
     *  Command-line-callable method for training and evaluating a relation tagger.
     *  <p>
     *  Takes 4 arguments:  training  test  documents  model <br>
     *  where  <br>
     *  training = file containing list of training documents  <br>
     *  test = file containing list of test documents  <br>
     *  documents = directory containing document files  <br>
     *  model = file to contain max ent model
     */
    
    public static void main (String[] args) throws IOException {
	if (args.length != 4) {
	    System.out.println ("Error, 4 arguments required:"); 
 	    System.out.println ("   listOfTrainingDocs listOfTestDocs documentDirectory modelFileName");
	    System.exit(1);                     
	} 
	String trainDocListFileName = args[0];
	String testDocListFileName = args[1];
	String docDir = args[2];
	String modelFN = args[3];
	Properties p = new Properties();
	p.setProperty("RelationTagger.model.fileName", modelFN);
    try(RelationTagger rtagger = new RelationTagger(p, true, TypeDetail.Basic)) {
        // rtagger.getExamples(docDir, trainDocListFileName, "training.txt");
        rtagger.trainTagger(docDir, trainDocListFileName);
        rtagger.evaluate(docDir, testDocListFileName);
    }
    }

    /**
     *  Train the relation tagger.
     *
     *  @param  docDir           directory containing document files
     *  @param  docListFileName  file containing list of training documents
     */

    public void trainTagger (String docDir, String docListFileName) throws IOException {
    if(useTFmodel) return; // Can't train TF model from here
	BufferedReader docListReader = new BufferedReader (new FileReader (docListFileName));
	PrintWriter eventWriter = new PrintWriter (new FileWriter ("events"));
        int docCount = 0;
	String line; 
	while ((line = docListReader.readLine()) != null) {
	    learnFromDocument (docDir + "/" + line.trim(), eventWriter);
            docCount++;
            if (docCount % 5 == 0) System.out.print(".");
        }
	eventWriter.close();
	model.train("events", 3);
    }
    public List<RelationExample> getExamples (String docDir, String docListFileName) throws IOException {
        return getExamples(docDir, docListFileName, null);
    }
    public List<RelationExample> getExamples (String docDir, String docListFileName, String outputName) throws IOException {
        BufferedReader docListReader = new BufferedReader (new FileReader (docListFileName));
        int docCount = 0;
        String line;
        List<RelationExample> examples = new ArrayList<RelationExample>();
        while ((line = docListReader.readLine()) != null) {
            examples.addAll(learnFromDocument(docDir + "/" + line.trim(), null));
            docCount++;
            if (docCount % 5 == 0 && printProgress) System.out.print(".");
        }
        if(outputName != null)
            Files.write(Paths.get(outputName), examples.stream()
                        .map(RelationExample::toString)
                        .collect(Collectors.toList()), Charset.forName("UTF-8"));
        return examples;
    }

    /**
     *  Acquire training data from one Document in the training corpus.
     *
     *  @param  docFileName  the name of the document file
     *  @param  eventWriter  the Writer onto which the feature vectors extracted from
     *                       the document are to be written
     */

    List<RelationExample> learnFromDocument (String docFileName, PrintWriter eventWriter) throws IOException {
	File docFile = new File(docFileName);
	Document doc = new Document(docFile);
	doc.setText(EntityTagger.eraseXML(doc.text()));
	String apfFileName = docFileName.replace("sgm" , "apf.xml");
	AceDocument aceDoc = new AceDocument(docFileName, apfFileName);
	// --- apply tokenizer and sentence segmenter
	Properties config = new Properties();
	config.setProperty("annotators", "token sentence");
	doc = Hub.processDocument(doc, config);
	// ---	
	findEntityMentions (aceDoc);
	findRelationMentions (aceDoc);
	// collect all pairs of nearby mentions
	List<AceEntityMention[]> pairs = findMentionPairs (doc);
	List<RelationExample> examples = new ArrayList<RelationExample>();
    // iterate over pairs of adjacent mentions, record candidates for ACE relations
    for (AceEntityMention[] pair : pairs) {
        RelationExample ex = addTrainingInstance (doc, pair[0], pair[1], eventWriter);
        if(ex != null) examples.add(ex);
    }
    return examples;
	// were any positive instances not captured?
	// reportLeftovers ();
    }

    static Set<AceEntityMention> mentionSet;

    /**
     *  Puts all mentions of all AceEntities in 'aceDoc' into 'mentionSet'.
     */

    static void findEntityMentions (AceDocument aceDoc) {
	mentionSet = new HashSet<AceEntityMention>();
	ArrayList<AceEntity> entities = aceDoc.entities;
	for (AceEntity entity : entities) {
	    String type = entity.type;
	    String subtype = entity.subtype;
	    ArrayList<AceEntityMention>  mentions = entity.mentions;
	    for (AceEntityMention mention : mentions) {
		mentionSet.add (mention);
	    }
	}
    }

    static List<AceRelationMention> relMentionList;

    /**
     *  Puts all AceRelationMentions in 'aceDoc' on relMentionList.
     */

    private static void findRelationMentions (AceDocument aceDoc) {
	relMentionList = new ArrayList<AceRelationMention>();
	ArrayList relations = aceDoc.relations;
	for (int i=0; i<relations.size(); i++) {
	    AceRelation relation = (AceRelation) relations.get(i);
	    String relationClass = relation.relClass;
	    relMentionList.addAll(relation.mentions);
	}
    }

    private static final int mentionWindow = 4;

    /**
     *  returns the set of all pairs of mentions separated by at most mentionWindow mentions
     */

    static List<AceEntityMention[]> findMentionPairs (Document doc) {
	List<AceEntityMention[]> pairs = new ArrayList<AceEntityMention[]> ();
	if (mentionSet.isEmpty()) return pairs;
	ArrayList mentionList = new ArrayList(mentionSet);
	Collections.sort(mentionList);
	for (int i=0; i<mentionList.size()-1; i++) {
	    for (int j=1; j<=mentionWindow && i+j<mentionList.size(); j++) {
		AceEntityMention m1 = (AceEntityMention) mentionList.get(i);
		AceEntityMention m2 = (AceEntityMention) mentionList.get(i+j);
		// if two mentions co-refer, they can't be in a relation
		// if (!canBeRelated(m1, m2)) continue;
		// if two mentions are not in the same sentence, they can't be in a relation
		if (!inSameSentence(m1.jetHead.start(), m2.jetHead.start(), doc)) continue;
		pairs.add(new AceEntityMention[] {m1, m2});
	    }
	}
	return pairs;
    }

    /**
     *  Returns true if character offsets <code>s1</code> and <code>s2</code>
     *  fall wihin the same sentence in Document doc.
     */

    static boolean inSameSentence (int s1, int s2, Document doc) {
	Vector<Annotation> sentences = doc.annotationsOfType("sentence");
	if (sentences == null) {
	    System.out.println("no sentence annotations");
	    return false;
	}
	for (Annotation sentence : sentences) 
	    if (within(s1,sentence.span()))
		return within(s2, sentence.span());
	return false;
    }

    private static boolean within (int i, Span s) {
	return (i >= s.start()) && (i <= s.end());}

    private String relationOutcome(AceEntityMention m1, AceEntityMention m2) {
    String outcome = OTHER;
loop:
    for (AceRelationMention mention : relMentionList) {
        if (mention.arg1 == m1 && mention.arg2 == m2) {
            switch(typeDetail) {
                case Basic:
                    outcome = mention.relation.type;
                    break;
                case Subtype:
                    outcome = mention.relation.type + ":" + mention.relation.subtype;
                    break;
                case SubtypeWithOrder:
                    outcome = mention.relation.type + ":" + mention.relation.subtype;
                    break;
            }
            relMentionList.remove(mention);
            break loop;
        } else if (mention.arg1 == m2 && mention.arg2 == m1) {
            switch(typeDetail) {
                case Basic:
                    outcome = mention.relation.type;
                    break;
                case Subtype:
                    outcome = mention.relation.type + ":" + mention.relation.subtype;
                    break;
                case SubtypeWithOrder:
                    outcome = mention.relation.type + ":" + mention.relation.subtype + "-1";
                    break;
            }
            relMentionList.remove(mention);
            break loop;
        }
    }
    return outcome;
    }
    /**
     *  Check whether there is a relation between m1 and m2 in the training corpus;
     *  If so, write the feature vector with the relation type (or, in the absence of a 
     *  relation, the outcome "other")).
     */

    private RelationExample addTrainingInstance (Document doc, AceEntityMention m1, AceEntityMention m2,
	    PrintWriter eventWriter) {
    String outcome = relationOutcome(m1, m2);
    if(useTFmodel) {
        List<WordInfo> infos = tfRelationFeatures(doc, m1, m2);
        return new RelationExample(infos, outcome);
    } else {
        Datum d = relationFeatures(doc, m1, m2);
        d.setOutcome(outcome);
        eventWriter.println(d);
        return null;
    }
    }
    private List<WordInfo> tfRelationFeatures (Document doc, AceEntityMention m1, AceEntityMention m2) {
        int ind = m1.jetHead.start();
        List<String> tokens = new ArrayList<String>();
        while(ind < m2.jetHead.end()) {
            Token t = doc.tokenAt(ind);
            if(t != null) {
                String text = doc.normalizedText(t).toLowerCase();
                if(TFRelationTagger.DIVIDE_TOKENS) for(String s : text.split(" "))
                    tokens.add(s);
                else tokens.add(text);
            }
            ind++;
        }
        int head1 = 0; // Assume 1st word in mention is head
        int head2 = tokens.size()-1; // Assume last word in mention is head
        if(tokens.size() > TFRelationTagger.INPUT_LEN) { // Length cutoff
            return null;
        } else if(tokens.size() < TFRelationTagger.INPUT_LEN) {
            int beg = m1.jetHead.start()-1;
            int end = m2.jetHead.end();
            int turn = 0;
            // end not fin, beg turn -> 0
            // end fin, beg turn -> 1
            // beg not fin, end turn -> 2
            // beg fin, end turn -> 3
            //
            while(tokens.size() != TFRelationTagger.INPUT_LEN) {
                if(turn == 0 || turn == 1) { // Time to add to beg
                    Token t = null;
                    while(t == null && beg > 0) {
                        t = doc.tokenAt(beg);
                        beg--;
                    }
                    if(t == null) { // Beg is fin
                        tokens.add(0, "emptytoken");
                        turn = 3;
                        head1++;
                        head2++;
                    } else {
                        String text = doc.normalizedText(t).toLowerCase();
                        if(TFRelationTagger.DIVIDE_TOKENS) {
                            String[] arr = text.split(" ");
                            for(int i = arr.length -1; i >= 0 && tokens.size() != TFRelationTagger.INPUT_LEN; i--) {
                                head1++;
                                head2++;
                                tokens.add(0, arr[i]);
                            }
                        } else {
                            tokens.add(0, text);
                            head1++;
                            head2++;
                        }
                        if(turn != 1) turn = 2;
                    }
                } else { // Time to add to end
                    Token t = null;
                    while(t == null && end < doc.length()) {
                        t = doc.tokenAt(end);
                        end++;
                    }
                    if(t == null) { // End is fin
                        tokens.add("emptytoken");
                        turn = 1;
                    } else {
                        String text = doc.normalizedText(t).toLowerCase();
                        if(TFRelationTagger.DIVIDE_TOKENS) {
                            String[] arr = text.split(" ");
                            for(int i = 0; i < arr.length && tokens.size() != TFRelationTagger.INPUT_LEN; i++) {
                                tokens.add(arr[i]);
                            }
                        } else tokens.add(text);
                        if(turn != 3) turn = 0;
                    }
                }
            }
        }
        List<WordInfo> infos = new ArrayList<WordInfo>();
        for(int i = 0; i < tokens.size(); i++) {
            infos.add(new WordInfo(tokens.get(i), i - head1, i - head2));
        }
        return infos;
    }

    /**
     *  Features for relation tagging:  the types and identities of the arguments
     *  and the number of words between the arguments.
     *  <p>
     *  There are two slightly different feature functions.  The first is used
     *  in training the tagger.  In training, we rely on 'perfect entity mentions'
     *  from the hand-tagged APF files.  The second is used in applying the
     *  tagger as part of a pipeline to process new text.  In that case we
     *  use entity mentions generated by prior stages in the pipeline
     */

    Datum relationFeatures (Document doc, AceEntityMention m1, AceEntityMention m2) {
	Datum d = new Datum(model);
	d.addFV ("arg1", m1.headText.replace(" ", "_").replace("\n", "_"));
	d.addFV ("arg2", m2.headText.replace(" ", "_").replace("\n", "_"));
	d.addFV ("type1", m1.entity.type);
	d.addFV ("type2", m2.entity.type);
	d.addFV ("types", m1.entity.type + "-" + m2.entity.type);
	int x = m1.jetHead.end();
	int wordsBetween = 0;
	String phraseBetween = "";
	while (x < m2.jetHead.start()) {
	    Token token = doc.tokenAt(x);
	    if (token == null) break;
	    String tokenText = doc.normalizedText(token);
	    wordsBetween++;
	    if (phraseBetween == "")
		phraseBetween = tokenText;
	    else
		phraseBetween += "_" + tokenText;
	     d.addF(doc.normalizedText(token));
	    x = token.end();
	}
	d.addFV ("WordsBetween", Integer.toString(wordsBetween));
	d.addFV ("phraseBetween", phraseBetween);
	return d;
    }

    /**
     *  Features for relation tagging:  the types and identities of the arguments
     *  and the number of words between the arguments.
     */

    Datum relationFeatures (Document doc, Mention m1, Mention m2) {
	Datum d = new Datum(model);
	d.addFV ("arg1", doc.normalizedText(m1));
	d.addFV ("arg2", doc.normalizedText(m2));
	String type1 = m1.getMentionOf().getSemType();
	String type2 = m2.getMentionOf().getSemType();
	d.addFV ("type1", type1);
	d.addFV ("type2", type2);
	d.addFV ("types", type1 + "-" + type2);
	int x = m1.end();
	int wordsBetween = 0;
	String phraseBetween = "";
	while (x < m2.start()) {
	    Annotation token = doc.tokenAt(x);
	    if (token == null) break;
	    String tokenText = doc.normalizedText(token);
	    wordsBetween++;
	    if (phraseBetween == "")
		phraseBetween = tokenText;
	    else
		phraseBetween += "_" + tokenText;
	    // d.addF(doc.normalizedText(token));
	    x = token.end();
	}
	d.addFV ("WordsBetween", Integer.toString(wordsBetween));
	// d.addFV ("phraseBetween", phraseBetween);
	return d;
    }

    static int correctRelations = 0;
    static int responseRelations = 0;
    static int keyRelations = 0;
    List<String> actuals;

    /**
     *  Evaluate the relation model just built and print the scores.
     *
     *  @param  docDir               directory containing test documents
     *  @param  testDocListFileName  file containing a list of test documents,
     *                               one per line
     */

    void evaluate (String docDir, String testDocListFileName) throws IOException {
	correctRelations = 0;
	responseRelations = 0;
	keyRelations = 0;
    actuals = new ArrayList<String>();
	BufferedReader docListReader = new BufferedReader (new FileReader (testDocListFileName));
	String line;
    List<List<WordInfo>> examples = new ArrayList<List<WordInfo>>();
	while ((line = docListReader.readLine()) != null)
	    examples.addAll(evaluateOnDocument(docDir + "/" + line.trim()));
    if(useTFmodel) {
        List<String> predictions = tfTagger.predictMultiple(examples);
        evaluatePredictions(predictions, actuals);
    }
	float recall = 100.0f * correctRelations / keyRelations;
	float precision = 100.0f * correctRelations / responseRelations;
	System.out.println ("correct: " + correctRelations + "   response: " + responseRelations
		+ "   key: " + keyRelations);
	float F = 2 * precision  * recall / (precision + recall);
	System.out.printf ( "  precision: %5.2f", precision);
	System.out.printf ( "  recall:    %5.2f",  recall);
	System.out.printf ( "  F1:        %5.2f \n",  F);
    } 

    /**
     *  Evaluate the model with respect to dcument 'docFileName' from the test collection.
     */

    List<List<WordInfo>> evaluateOnDocument (String docFileName) throws IOException {
	File docFile = new File(docFileName);
	Document doc = new Document(docFile);
	doc.setText(EntityTagger.eraseXML(doc.text()));
	String apfFileName = docFileName.replace("sgm" , "apf.xml");
	AceDocument aceDoc = new AceDocument(docFileName, apfFileName);
	// --- apply tokenizer and sentence segmenter
	Properties config = new Properties();
	config.setProperty("annotators", "token sentence");
    doc = Hub.processDocument(doc, config);
	// ---
	findEntityMentions (aceDoc);
	findRelationMentions (aceDoc);
	// collect all pairs of nearby mentions
	List<AceEntityMention[]> pairs = findMentionPairs (doc);
	// iterate over pairs of adjacent mentions, record candidates for ACE relations
    if(useTFmodel) {
        List<List<WordInfo>> testExamples = new ArrayList<List<WordInfo>>();
        for (AceEntityMention[] pair : pairs) {
            List<WordInfo> example = tfRelationFeatures(doc, pair[0], pair[1]);
            if(example != null) {
                testExamples.add(example);
                actuals.add(relationOutcome(pair[0], pair[1]));
            } else {
                evaluatePrediction(OTHER, relationOutcome(pair[0], pair[1]));
            }
        }
        return testExamples;
    } else {
        for (AceEntityMention[] pair : pairs) evaluateOnPair (doc, pair[0], pair[1]);
            return null;
    }
    }
	void evaluatePredictions (List<String> predictions, List<String> actuals) {
        for(int i = 0; i < predictions.size(); i++) {
            String prediction = predictions.get(i);
            String actual = actuals.get(i);
            evaluatePrediction(prediction, actual);
        }
    }
    void evaluatePrediction (String prediction, String actual) {
        if(prediction.equals(actual) && !prediction.equals(OTHER)) correctRelations++;
        if(!prediction.equals(OTHER)) responseRelations ++;
        if(!actual.equals(OTHER)) keyRelations++;
    }

    /**
     *  Evaluate the relation tagger with respect to a specific pair of entity mentions.
     */
    void evaluateOnPair (Document doc, AceEntityMention m1, AceEntityMention m2) {
	// generate features and predict relation
    Datum d = relationFeatures(doc, m1, m2);
    String prediction = model.getBestOutcome(d.toArray());
	// determine from ACE key whether there is a relation
	String outcome = relationOutcome(m1, m2);
	evaluatePrediction(prediction, outcome);
    }

    /**
     *  Annotate a document with RelationMention annotations.  
     */

    public Document annotate (Document doc, Span span) {
	// load model if not previously loaded.
	if (!model.isLoaded())
	    model.loadModel();
	List<Mention> mentionList = Coref.gatherMentions(doc, span);
	// iterate over all pairs of entity mentions appearing in the same sentence
	for (int i=0; i<mentionList.size()-1; i++) {
	    for (int j=1; j<=mentionWindow && i+j<mentionList.size(); j++) {
		Mention m1 = mentionList.get(i);
		Mention m2 = mentionList.get(i+j);
		// if two mentions are not in the same sentence, they can't be in a relation
		if (!inSameSentence(m1.start(), m2.start(), doc)) continue;
		// compte the features for this mentin pair and then use the
		// Maxent model to predict the relation, if any
		Datum d = relationFeatures (doc, m1, m2);
		String prediction = model.getBestOutcome(d.toArray());
		// if model predicts a relation, add a RelationMention annotation
		if ( !prediction.equals(OTHER)) {
		    Span relSpan;
		    if (m1.start() < m2.start())
			relSpan = new Span (m1.start(), m2.end());
		    else
			relSpan = new Span (m2.start(), m1.end());
		    RelationMention rm = new RelationMention(relSpan);
		    doc.addAnnotation(rm);
		    rm.setSemType(prediction);
		    System.out.println("* Found relation " + doc.normalizedText(relSpan));
		    System.out.println("  arg1= " + doc.normalizedText(m1) +
			    " type = " + prediction + " arg2 = " + doc.normalizedText(m2));
		}
	    }
	}
	return doc;
    }

}
