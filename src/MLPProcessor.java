import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;

public class MLPProcessor {

    public static final String datasetFile = "vehicledata.arff";
    public static final String testDataFile = "testdata.arff";


    public MLPProcessor() {
        try {
            FileReader fileReader = new FileReader(datasetFile);

            Instances training = new Instances(fileReader);
            training.setClassIndex(training.numAttributes() - 1);

            MultilayerPerceptron mlp = new MultilayerPerceptron();

            mlp.setOptions(Utils.splitOptions("-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 4"));
            mlp.buildClassifier(training);

            FileReader tr = new FileReader(testDataFile);
            Instances testdata = new Instances(tr);

            testdata.setClassIndex(testdata.numAttributes() - 1);

            Evaluation eval = new Evaluation(training);
            eval.evaluateModel(mlp, testdata);

            System.out.println(eval.toSummaryString("\nResultado\n*******\n", false));

            tr.close();
            fileReader.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        MLPProcessor mlp = new MLPProcessor();
    }
}
