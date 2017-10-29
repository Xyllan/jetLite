package edu.nyu.jetlite;
/**
 * Created by xuanwang on 10/17/17.
 * https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
 */
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class TFRelationTagger implements AutoCloseable {
    public static String MODEL_DIR = "RTFmodel/";
    public static String DATA_DIR = "data/";
    public static Path PRETRAINED_WORD_EMBEDDINGS_PATH = Paths.get(DATA_DIR, "glove.6B.300d.txt");
    
    public static int INPUT_LEN = 15; // Max distance between two entity heads for an example to be considered.
    public static int POS_EMBED_LEN = INPUT_LEN * 2 - 1;
    public static boolean DIVIDE_TOKENS = true; // If true, divides tokens with spaces into separate vars.
    public static int WORD_EMBED_LEN = 300; // # of dims in word embedding
    public static int WORD_FEAT_LEN = WORD_EMBED_LEN + 2; // per word features: word embedding + 2 positional inputs

    public static String[] BASIC_LABELS = new String[] { "OTHER", "GEN-AFF", "ORG-AFF",
        "PART-WHOLE", "PER-SOC", "PHYS", "ART" };

    public static String[] SUBTYPE_LABELS = new String[] { "OTHER", "GEN-AFF:Citizen-Resident-Religion-Ethnicity",
        "ORG-AFF:Employment", "PART-WHOLE:Subsidiary", "ORG-AFF:Membership", "ORG-AFF:Ownership", "PER-SOC:Business",
        "GEN-AFF:Org-Location", "PHYS:Located", "PART-WHOLE:Geographical", "ORG-AFF:Founder",
        "ART:User-Owner-Inventor-Manufacturer", "PHYS:Near", "PER-SOC:Family", "PART-WHOLE:Artifact",
        "PER-SOC:Lasting-Personal", "ORG-AFF:Student-Alum", "ORG-AFF:Investor-Shareholder", "ORG-AFF:Sports-Affiliation" };

    public static String[] SUBTYPE_WITH_ORDER_LABELS = new String[] { "OTHER", "GEN-AFF:Citizen-Resident-Religion-Ethnicity",
        "ORG-AFF:Employment-1", "ORG-AFF:Employment", "GEN-AFF:Citizen-Resident-Religion-Ethnicity-1",
        "PART-WHOLE:Subsidiary-1", "ORG-AFF:Membership", "ORG-AFF:Ownership", "PER-SOC:Business",
        "GEN-AFF:Org-Location", "PHYS:Located-1", "PHYS:Located", "PART-WHOLE:Geographical", "ORG-AFF:Founder-1",
        "ORG-AFF:Membership-1", "PART-WHOLE:Geographical-1", "ART:User-Owner-Inventor-Manufacturer", "PHYS:Near",
        "ART:User-Owner-Inventor-Manufacturer-1", "PART-WHOLE:Subsidiary", "PHYS:Near-1", "PER-SOC:Family",
        "GEN-AFF:Org-Location-1", "PER-SOC:Family-1", "PART-WHOLE:Artifact-1", "PER-SOC:Business-1",
        "PART-WHOLE:Artifact", "PER-SOC:Lasting-Personal", "ORG-AFF:Student-Alum-1", "ORG-AFF:Founder",
        "ORG-AFF:Student-Alum", "ORG-AFF:Ownership-1", "ORG-AFF:Investor-Shareholder", "ORG-AFF:Investor-Shareholder-1",
        "ORG-AFF:Sports-Affiliation", "ORG-AFF:Sports-Affiliation-1", "PER-SOC:Lasting-Personal-1" };
    
    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    public static float[] convertToFloatArray(double[] doubleArray) {
        float[] floatArray = new float[doubleArray.length];
        for(int i = 0; i < doubleArray.length; i++) floatArray[i] = (float) doubleArray[i];
        return floatArray;
    }

    public static Map<String, float[]> getAllEmbeddings() {
        Map<String, float[]> tokens = new HashMap<String, float[]>();
        try(Stream<String> stream = Files.lines(PRETRAINED_WORD_EMBEDDINGS_PATH)) {
            stream.forEach(l -> {
                String[] data = l.split(" ");
                double[] vals = Arrays.stream(data).skip(1)
                            .mapToDouble(Double::parseDouble)
                            .toArray();
                tokens.put(data[0], convertToFloatArray(vals));
            });
            return tokens;
        } catch(IOException e) {
            System.out.println(e);
        } catch(Exception e) {
            System.out.println(e);
        }
        return null;
    }

    public static float[] UNKNOWN_WORD_EMBEDDING = new float[WORD_EMBED_LEN];

    // The detail with which to classify a relation
    RelationTagger.TypeDetail typeDetail;

    // Number of outcome classes
    int numClasses;

    // Word embeddings
    Map<String, float[]> wordEmbeddings;

    // Class ID to class name mapping
    Map<Integer, String> classMapping;

    // Model path
    Path modelPath;

    // Graph Definiton
    byte[] graphDef;

    public TFRelationTagger(RelationTagger.TypeDetail typeDetail) {
        this.typeDetail = typeDetail;
        String[] labels;
        switch(typeDetail) {
            case Basic:
            default:
                numClasses = 7;
                labels = BASIC_LABELS;
                modelPath = Paths.get(MODEL_DIR, "frozen_basic.pb");
                break;
            case Subtype:
                numClasses = 19;
                labels = SUBTYPE_LABELS;
                modelPath = Paths.get(MODEL_DIR, "frozen_subtype.pb");
                break;
            case SubtypeWithOrder:
                numClasses = 37;
                labels = SUBTYPE_WITH_ORDER_LABELS;
                modelPath = Paths.get(MODEL_DIR, "frozen_subtype_with_order.pb");
                break;
        }
        graphDef = readAllBytesOrExit(modelPath);
        classMapping = IntStream.range(0, labels.length).boxed().collect(Collectors.toMap(i -> i, i -> labels[i]));
        wordEmbeddings = getAllEmbeddings();
    }

    private float[] getEmbedding(String word) {
        float[] emb = wordEmbeddings.get(word);
        if(emb != null) {
            return emb;
        } else {
            return UNKNOWN_WORD_EMBEDDING;
        }
    }

    // Convert a single example into a float tensor for graph input
    public float[][][] convertSingle(List<WordInfo> example) {
        float[][][] input = new float[1][INPUT_LEN][WORD_EMBED_LEN + 2];
        for(int j = 0; j < example.size(); j++) {
            WordInfo info = example.get(j);
            float[] vals = wordEmbeddings.get(info.word);
            int k;
            for(k = 0; k < vals.length; k++) {
                input[0][j][k] = vals[k];
            }
            input[0][j][k] = info.distance1;
            input[0][j][k+1] = info.distance2;
        }
        return input;
    }

    // Convert a list of examples into a float tensor for graph input
    public float[][][] convertMultiple(List<List<WordInfo>> examples) {
        float[][][] input = new float[examples.size()][INPUT_LEN][WORD_EMBED_LEN + 2];
        for(int i = 0; i < examples.size(); i++) {
            List<WordInfo> example = examples.get(i);
            for(int j = 0; j < example.size(); j++) {
                WordInfo info = example.get(j);
                float[] vals = getEmbedding(info.word);
                int k;
                for(k = 0; k < vals.length; k++) {
                    input[i][j][k] = vals[k];
                }
                input[i][j][k] = info.distance1;
                input[i][j][k+1] = info.distance2;
            }
        }
        return input;
    }

    // Predict a single example
    public String predictSingle(List<WordInfo> example) {
        float[][][] input = convertSingle(example);
        return predict(input).get(0);
    }

    // Predict a list of examples
    public List<String> predictMultiple(List<List<WordInfo>> examples) {
        float[][][] input = convertMultiple(examples);
        return predict(input);
    }

    private List<String> predict(float[][][] input) {
        float[][] res = runGraph(input);
        List<String> predictions = new ArrayList<String>();
        for(int i = 0; i < res.length; i++) {
            predictions.add(classMapping.get(maxIndex(res[i])));
        }
        return predictions;
    }

    private float[][] runGraph(float[][][] input) {
        try(Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try(Session s = new Session(g);
                Tensor inputTensor = Tensor.create(input);
                Tensor keepProb = Tensor.create(1f);
                Tensor result = s.runner().feed("input", inputTensor).feed("keep_prob", keepProb).fetch("output").run().get(0)) {
                return result.copyTo(new float[input.length][numClasses]);
            }
        }
    }

    public void close() { }

    public static void main(String[] args) {
        Random generator = new Random();
        int nExamples = 1;
        int seqLen = 15;
        int featLen = 302;
        float[][][] input = new float[nExamples][seqLen][featLen];
        for(int i = 0; i < nExamples; i++) {
            for(int j = 0; j < seqLen; j++) {
                for(int k = 0; k < featLen; k++) {
                    input[i][j][k] = generator.nextFloat();
                }
            }
        }

        // preprocess the image to feed into inception model
        Tensor keepProb = Tensor.create(1f);
        byte[] graphDef = readAllBytesOrExit(Paths.get(MODEL_DIR, "frozen_basic.pb"));
        float[][] output = executeGraph(graphDef, createInput(input), keepProb);
        for(int i = 0; i < nExamples; i++) {
            System.out.println(maxIndex(output[i]));
        }
    }

    private static Tensor createInput(float[][][] input) {
        try (Graph g = new Graph()) {
            GraphBuilder b = new GraphBuilder(g);
            // Since the graph is being constructed once per execution here, we can use a constant for the
            // input image. If the graph were to be re-used for multiple input images, a placeholder would
            // have been more appropriate.
            final Output output = b.constant("input", input);
            try (Session s = new Session(g)) {
                return s.runner().fetch(output.op().name()).run().get(0);
            }
        }
    }

    private static float[][] executeGraph(byte[] graphDef, Tensor image, Tensor keepProb) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                Tensor result = s.runner().feed("input", image).feed("keep_prob", keepProb).fetch("output").run().get(0)) {
                return result.copyTo(new float[1][7]);
            }
        }
    }
    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    // In the fullness of time, equivalents of the methods of this class should be auto-generated from
    // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
    // like Python, C++ and Go.
    static class GraphBuilder {
        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output constant(String name, Object value) {
            try (Tensor t = Tensor.create(value)) {
                return g.opBuilder("Const", name)
                        .setAttr("dtype", t.dataType())
                        .setAttr("value", t)
                        .build()
                        .output(0);
            }
        }

        private Graph g;
    }
}
