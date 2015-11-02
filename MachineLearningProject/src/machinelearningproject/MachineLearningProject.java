/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */ 

package machinelearningproject;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Raches
 */
public class MachineLearningProject {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        DataSource source = new DataSource("D:\\spambase.arff");
//        DataSource source = new DataSource("D:\\weather-nominal.arff");
        Instances instances = source.getDataSet();
        int numAttr = instances.numAttributes();
        instances.setClassIndex(instances.numAttributes()-1);

        Random rand_ = new Random();  
        instances.randomize(rand_);
        
        Random randeval = new Random(1);
        Evaluation eval = new Evaluation(instances);
        
        // ID3 Evaluation
//        System.out.println("\n=== ID3 EVALUATION ===");  
//        DecisionTree id3 = new DecisionTree();
//        
//        eval.crossValidateModel(id3, instances, 10, randeval); 
//        System.out.println(eval.toSummaryString());
//        System.out.println(eval.toClassDetailsString());
//        System.out.println(eval.toMatrixString());
//        
//        // Random Forest Evaluation
//        System.out.println("\n=== RANDOM FOREST EVALUATION ===");
//        
//        RandomForest rf = new RandomForest(5);
//        
//        eval.crossValidateModel(rf, instances, 10, randeval); 
//        System.out.println(eval.toSummaryString());
//        System.out.println(eval.toClassDetailsString());
//        System.out.println(eval.toMatrixString());
        
        // SVM Evaluation
//        ArrayList<Double> list = new ArrayList<Double>(Arrays.asList(0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50));
//        ArrayList<Double> list = new ArrayList<Double>(Arrays.asList(0.55,0.60,0.65,0.70,0.75,0.80,0.95,1.00));
//        for (int i = 0; i < list.size(); i++){
            System.out.println("\n==== SVM EVALUATION ====");
            SMO svm = new SMO();
            RBFKernel rbfKernel = new RBFKernel();
            double gamma = 0.50;
            rbfKernel.setGamma(gamma);

            svm.setKernel(rbfKernel);
            long starttime = System.currentTimeMillis(); 
            svm.buildClassifier(instances);
            long stoptime = System.currentTimeMillis();
            long elapsedtime = stoptime - starttime;
            System.out.println(" === ELAPSED TIME "+ elapsedtime/1000 +" seconds ===");
            eval.crossValidateModel(svm, instances, 10, randeval);
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
//        }
    }
    
}
